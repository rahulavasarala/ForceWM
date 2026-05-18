#include "utils.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

namespace {

constexpr mjtNum kRenderTimestep = 1.0 / 60.0;
constexpr const char* kDefaultSceneXmlPath = "models/parametric_scene.xml";
constexpr const char* kDefaultRobotUrdfPath = "models/fr3.urdf";

}  // namespace

std::filesystem::path repo_root_directory() {
  return std::filesystem::path(FORCEWM_MODEL_ROOT).parent_path();
}

std::filesystem::path default_contract_path() {
  return repo_root_directory() / "universal_contract.yaml";
}

std::filesystem::path resolve_input_path(const char* argument) {
  if (!argument) {
    return default_contract_path();
  }

  const std::filesystem::path input_path(argument);
  if (input_path.is_absolute()) {
    return input_path.lexically_normal();
  }

  return std::filesystem::absolute(input_path).lexically_normal();
}

std::filesystem::path resolve_contract_relative_path(
    const std::filesystem::path& contract_path,
    const std::string& path_string) {
  const std::filesystem::path input_path(path_string);
  if (input_path.is_absolute()) {
    return input_path;
  }
  return (contract_path.parent_path() / input_path).lexically_normal();
}

std::string normalize_mode_token(std::string mode) {
  std::transform(mode.begin(), mode.end(), mode.begin(),
                 [](unsigned char character) {
                   if (character == '-' || character == ' ') {
                     return static_cast<char>('_');
                   }
                   return static_cast<char>(std::tolower(character));
                 });
  return mode;
}

bool parse_mode_argument(const std::string& mode_argument,
                         bool& parsed_is_data_collection) {
  const std::string normalized_mode = normalize_mode_token(mode_argument);
  if (normalized_mode == "inference") {
    parsed_is_data_collection = false;
    return true;
  }

  if (normalized_mode == "data_collection" ||
      normalized_mode == "datacollection" ||
      normalized_mode == "collection") {
    parsed_is_data_collection = true;
    return true;
  }

  return false;
}

void print_usage(const char* executable_name) {
  std::cout << "Usage: " << executable_name
            << " [universal_contract.yaml] [inference|data_collection]\n"
            << "Defaults are loaded from " << default_contract_path().string()
            << " and mode defaults to inference.\n"
            << "\n"
            << "Examples:\n"
            << "  " << executable_name << "\n"
            << "  " << executable_name << " data_collection\n"
            << "  " << executable_name
            << " /full/path/to/universal_contract.yaml inference\n";
}

std::optional<StartupOptions> parse_startup_options(int argc, char** argv) {
  if (argc > 3) {
    print_usage(argv[0]);
    return std::nullopt;
  }

  StartupOptions options;
  const char* contract_argument = nullptr;
  const char* mode_argument = nullptr;

  if (argc == 2) {
    if (!parse_mode_argument(argv[1], options.is_data_collection)) {
      contract_argument = argv[1];
    }
  } else if (argc == 3) {
    contract_argument = argv[1];
    mode_argument = argv[2];
  }

  if (mode_argument &&
      !parse_mode_argument(mode_argument, options.is_data_collection)) {
    std::cerr << "Invalid mode `" << mode_argument
              << "`. Expected `inference` or `data_collection`.\n";
    print_usage(argv[0]);
    return std::nullopt;
  }

  options.contract_path = resolve_input_path(contract_argument);
  return options;
}

std::string require_string(const YAML::Node& parent,
                           const char* key,
                           const std::string& context) {
  const YAML::Node value = parent[key];
  if (!value) {
    throw std::runtime_error("Missing required `" + std::string(key) +
                             "` in " + context + ".");
  }

  const std::string parsed_value = value.as<std::string>();
  if (parsed_value.empty()) {
    throw std::runtime_error("Required `" + std::string(key) + "` in " +
                             context + " cannot be empty.");
  }

  return parsed_value;
}

std::string make_redis_key(const std::string& prefix,
                           const std::string& suffix) {
  if (prefix.empty()) {
    return suffix;
  }
  return prefix + "::" + suffix;
}

std::string lowercase_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  return value;
}

VisualStreamType parse_visual_stream_type(const std::string& visual_type,
                                          const std::string& context) {
  const std::string normalized_type = lowercase_copy(visual_type);
  if (normalized_type == "rgb") {
    return VisualStreamType::kRgb;
  }
  if (normalized_type == "depth") {
    return VisualStreamType::kDepth;
  }

  throw std::runtime_error(
      "Unsupported visual `type: " + visual_type + "` in " + context +
      ". simulation.cpp only supports `rgb` and `depth` visual streams.");
}

SimulationContractConfig load_simulation_contract(
    const std::filesystem::path& contract_path) {
  const YAML::Node contract = YAML::LoadFile(contract_path.string());
  const YAML::Node robot_cfg = contract["robot"];
  if (!robot_cfg || !robot_cfg.IsMap()) {
    throw std::runtime_error("Expected a top-level `robot` mapping in " +
                             contract_path.string() + ".");
  }

  const std::string contract_context =
      "contract `" + contract_path.string() + "`";
  const std::string robot_type =
      require_string(robot_cfg, "type", contract_context);
  if (robot_type != "sim") {
    throw std::runtime_error(
        "simulation.cpp only supports `robot.type: sim`, but the contract sets "
        "`robot.type: " +
        robot_type + "`.");
  }

  SimulationContractConfig config;
  config.prefix = node_or<std::string>(robot_cfg["prefix"], "");
  config.xml_path = resolve_contract_relative_path(
      contract_path,
      node_or<std::string>(robot_cfg["xml_path"], kDefaultSceneXmlPath));
  config.urdf_path = resolve_contract_relative_path(
      contract_path,
      node_or<std::string>(robot_cfg["urdf_path"], kDefaultRobotUrdfPath));

  const YAML::Node visual_cfg =
      robot_cfg["data_sources"] ? robot_cfg["data_sources"]["visual"]
                                : YAML::Node();
  const double default_camera_fps =
      node_or<double>(visual_cfg["fps"], 1.0 / kRenderTimestep);
  const YAML::Node camera_keys = visual_cfg ? visual_cfg["keys"] : YAML::Node();

  if (camera_keys && !camera_keys.IsSequence()) {
    throw std::runtime_error(
        "`robot.data_sources.visual.keys` must be a sequence in " +
        contract_path.string() + ".");
  }

  if (!camera_keys) {
    return config;
  }

  std::vector<VisualContractEntry> visual_entries;
  visual_entries.reserve(camera_keys.size());
  size_t depth_stream_count = 0;

  for (const auto& camera_entry : camera_keys) {
    if (!camera_entry.IsMap() || camera_entry.size() != 1) {
      throw std::runtime_error(
          "Each entry in `robot.data_sources.visual.keys` must be a single-key "
          "mapping in " +
          contract_path.string() + ".");
    }

    const auto camera_it = camera_entry.begin();
    const std::string visual_name = camera_it->first.as<std::string>();
    const YAML::Node camera_cfg = camera_it->second;
    const std::string camera_context =
        "camera `" + visual_name + "` in " + contract_context;

    VisualContractEntry visual_entry;
    visual_entry.visual_name = visual_name;
    const std::string redis_suffix =
        node_or<std::string>(camera_cfg["redis"], visual_name);
    visual_entry.redis_key = make_redis_key(config.prefix, redis_suffix);
    visual_entry.metadata_redis_key = visual_entry.redis_key + "::meta";
    visual_entry.type = parse_visual_stream_type(
        node_or<std::string>(camera_cfg["type"], "rgb"), camera_context);
    visual_entry.source =
        lowercase_copy(node_or<std::string>(camera_cfg["source"], "sim"));
    visual_entry.encoding = lowercase_copy(node_or<std::string>(
        camera_cfg["encoding"],
        visual_entry.type == VisualStreamType::kDepth ? "png16" : "jpeg"));
    visual_entry.fps = node_or<double>(camera_cfg["fps"], default_camera_fps);
    visual_entry.has_explicit_fps = static_cast<bool>(camera_cfg["fps"]);

    if (visual_entry.source != "sim") {
      throw std::runtime_error(
          "simulation.cpp only supports `source: sim`, but camera `" +
          visual_name + "` sets `source: " + visual_entry.source + "`.");
    }

    const YAML::Node dim_cfg = camera_cfg["dim"];
    if (dim_cfg && dim_cfg.IsSequence()) {
      if (dim_cfg.size() >= 2) {
        visual_entry.width = dim_cfg[0].as<int>();
        visual_entry.height = dim_cfg[1].as<int>();
        visual_entry.has_explicit_dimensions = true;
      }
      if (dim_cfg.size() >= 3) {
        visual_entry.channels = dim_cfg[2].as<int>();
        visual_entry.has_explicit_channels = true;
      }
    }

    if (visual_entry.fps <= 0.0) {
      throw std::runtime_error("Camera '" + visual_name +
                               "' has a non-positive fps in " +
                               contract_path.string() + ".");
    }

    if (visual_entry.type == VisualStreamType::kRgb) {
      visual_entry.channels = visual_entry.has_explicit_channels
                                  ? visual_entry.channels
                                  : 3;
      if (visual_entry.channels != 3) {
        throw std::runtime_error("Camera '" + visual_name + "' requests " +
                                 std::to_string(visual_entry.channels) +
                                 " channels, but simulation publishing only "
                                 "supports RGB cameras right now.");
      }
      if (visual_entry.encoding != "jpeg") {
        throw std::runtime_error("Camera '" + visual_name +
                                 "' requests encoding `" +
                                 visual_entry.encoding +
                                 "`, but simulation publishing only supports "
                                 "JPEG for RGB streams right now.");
      }
      visual_entry.mujoco_camera_name =
          require_string(camera_cfg, "mujoco_camera_name", camera_context);
    } else {
      ++depth_stream_count;
      visual_entry.channels = visual_entry.has_explicit_channels
                                  ? visual_entry.channels
                                  : 1;
      if (visual_entry.channels != 1) {
        throw std::runtime_error("Depth stream '" + visual_name +
                                 "' requests " +
                                 std::to_string(visual_entry.channels) +
                                 " channels, but simulation depth publishing "
                                 "only supports single-channel depth.");
      }
      if (visual_entry.encoding != "png16") {
        throw std::runtime_error("Depth stream '" + visual_name +
                                 "' requests encoding `" +
                                 visual_entry.encoding +
                                 "`, but simulation depth publishing only "
                                 "supports png16.");
      }
      visual_entry.unit =
          lowercase_copy(require_string(camera_cfg, "unit", camera_context));
      if (visual_entry.unit != "mm") {
        throw std::runtime_error("Depth stream '" + visual_name +
                                 "' requests unit `" + visual_entry.unit +
                                 "`, but simulation depth publishing only "
                                 "supports millimeters (`mm`).");
      }
      visual_entry.align_to_visual_name =
          require_string(camera_cfg, "align_to", camera_context);
    }

    visual_entries.push_back(std::move(visual_entry));
  }

  if (depth_stream_count > 1) {
    throw std::runtime_error(
        "simulation.cpp currently supports at most one aligned depth stream.");
  }

  std::unordered_map<std::string, const VisualContractEntry*>
      visual_entry_by_name;
  visual_entry_by_name.reserve(visual_entries.size());
  for (const auto& visual_entry : visual_entries) {
    const auto [entry_it, inserted] =
        visual_entry_by_name.emplace(visual_entry.visual_name, &visual_entry);
    if (!inserted) {
      throw std::runtime_error("Duplicate visual key `" +
                               visual_entry.visual_name + "` in " +
                               contract_path.string() + ".");
    }
  }

  std::unordered_map<std::string, DepthStreamConfig> aligned_depth_by_rgb_key;
  for (const auto& visual_entry : visual_entries) {
    if (visual_entry.type != VisualStreamType::kDepth) {
      continue;
    }

    const auto aligned_rgb_it =
        visual_entry_by_name.find(visual_entry.align_to_visual_name);
    if (aligned_rgb_it == visual_entry_by_name.end()) {
      throw std::runtime_error("Depth stream `" + visual_entry.visual_name +
                               "` aligns to `" +
                               visual_entry.align_to_visual_name +
                               "`, but that RGB stream was not found in " +
                               contract_path.string() + ".");
    }

    const VisualContractEntry& aligned_rgb_entry = *aligned_rgb_it->second;
    if (aligned_rgb_entry.type != VisualStreamType::kRgb) {
      throw std::runtime_error("Depth stream `" + visual_entry.visual_name +
                               "` aligns to `" +
                               visual_entry.align_to_visual_name +
                               "`, but that stream is not RGB.");
    }
    if (visual_entry.has_explicit_fps &&
        std::abs(visual_entry.fps - aligned_rgb_entry.fps) > 1e-9) {
      throw std::runtime_error("Depth stream `" + visual_entry.visual_name +
                               "` must reuse the aligned RGB fps (" +
                               std::to_string(aligned_rgb_entry.fps) +
                               "), but it requests " +
                               std::to_string(visual_entry.fps) + ".");
    }

    DepthStreamConfig depth_stream;
    depth_stream.visual_name = visual_entry.visual_name;
    depth_stream.redis_key = visual_entry.redis_key;
    depth_stream.metadata_redis_key = visual_entry.metadata_redis_key;
    depth_stream.encoding = visual_entry.encoding;
    depth_stream.unit = visual_entry.unit;
    depth_stream.align_to_visual_name = visual_entry.align_to_visual_name;
    depth_stream.width = visual_entry.has_explicit_dimensions
                             ? visual_entry.width
                             : aligned_rgb_entry.width;
    depth_stream.height = visual_entry.has_explicit_dimensions
                              ? visual_entry.height
                              : aligned_rgb_entry.height;
    depth_stream.channels = visual_entry.channels;

    if (depth_stream.width != aligned_rgb_entry.width ||
        depth_stream.height != aligned_rgb_entry.height) {
      throw std::runtime_error("Depth stream `" + visual_entry.visual_name +
                               "` must match the aligned RGB dimensions (" +
                               std::to_string(aligned_rgb_entry.width) + "x" +
                               std::to_string(aligned_rgb_entry.height) +
                               "), but it requests " +
                               std::to_string(depth_stream.width) + "x" +
                               std::to_string(depth_stream.height) + ".");
    }

    if (!aligned_depth_by_rgb_key
             .emplace(visual_entry.align_to_visual_name,
                      std::move(depth_stream))
             .second) {
      throw std::runtime_error(
          "simulation.cpp only supports one depth stream aligned to an RGB key.");
    }
  }

  for (const auto& visual_entry : visual_entries) {
    if (visual_entry.type != VisualStreamType::kRgb) {
      continue;
    }

    CameraStreamConfig camera;
    camera.visual_name = visual_entry.visual_name;
    camera.redis_key = visual_entry.redis_key;
    camera.metadata_redis_key = visual_entry.metadata_redis_key;
    camera.mujoco_camera_name = visual_entry.mujoco_camera_name;
    camera.encoding = visual_entry.encoding;
    camera.type = visual_entry.type;
    camera.width = visual_entry.width;
    camera.height = visual_entry.height;
    camera.channels = visual_entry.channels;
    camera.fps = visual_entry.fps;

    const size_t image_size =
        static_cast<size_t>(camera.width) * camera.height * camera.channels;
    camera.rgb_buffer.resize(image_size);
    camera.flipped_rgb_buffer.resize(image_size);
    camera.bgr_buffer.resize(image_size);

    const auto aligned_depth_it =
        aligned_depth_by_rgb_key.find(camera.visual_name);
    if (aligned_depth_it != aligned_depth_by_rgb_key.end()) {
      DepthStreamConfig depth_stream = std::move(aligned_depth_it->second);
      const size_t depth_pixel_count =
          static_cast<size_t>(depth_stream.width) * depth_stream.height;
      depth_stream.raw_depth_buffer.resize(depth_pixel_count);
      depth_stream.flipped_depth_buffer.resize(depth_pixel_count);
      depth_stream.depth_mm_buffer.resize(depth_pixel_count);
      camera.aligned_depth = std::move(depth_stream);
    }

    config.cameras.push_back(std::move(camera));
  }

  return config;
}
