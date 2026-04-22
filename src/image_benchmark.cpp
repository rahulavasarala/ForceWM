#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "redis/RedisClient.h"

#include <zmq.h>

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct BenchmarkOptions {
  std::size_t frame_count = 100;
  int width = 224;
  int height = 224;
  int jpeg_quality = 90;
  std::string redis_namespace = "sai";
  std::string redis_host = "127.0.0.1";
  int redis_port = 6379;
  std::string redis_key = "benchmark::image";
  std::string zmq_endpoint = "tcp://127.0.0.1:5557";
  int zmq_timeout_ms = 5000;
};

struct PayloadStats {
  std::size_t total_bytes = 0;
  std::size_t min_bytes = 0;
  std::size_t max_bytes = 0;
  double average_bytes = 0.0;
};

struct BenchmarkResult {
  std::string transport_name;
  std::size_t frame_count = 0;
  std::size_t total_bytes = 0;
  double elapsed_seconds = 0.0;
};

void print_usage(const char* program_name) {
  std::cout
      << "Usage: " << program_name << " [options]\n"
      << "  --frames <count>          Number of frames to send (default: 100)\n"
      << "  --width <pixels>          Frame width before JPEG encoding (default: 224)\n"
      << "  --height <pixels>         Frame height before JPEG encoding (default: 224)\n"
      << "  --jpeg-quality <0-100>    JPEG quality used to build payloads (default: 90)\n"
      << "  --redis-namespace <name>  Redis namespace prefix (default: sai)\n"
      << "  --redis-host <host>       Redis host (default: 127.0.0.1)\n"
      << "  --redis-port <port>       Redis port (default: 6379)\n"
      << "  --redis-key <key>         Redis key for frame writes (default: benchmark::image)\n"
      << "  --zmq-endpoint <endpoint> ZMQ REP endpoint to bind/connect (default: tcp://127.0.0.1:5557)\n"
      << "  --zmq-timeout-ms <ms>     Send/recv timeout for ZMQ sockets (default: 5000)\n"
      << "  --help                    Show this message\n";
}

int parse_positive_int(const std::string& value, const std::string& option_name) {
  std::size_t parsed_chars = 0;
  int parsed_value = 0;
  try {
    parsed_value = std::stoi(value, &parsed_chars);
  } catch (const std::exception&) {
    throw std::runtime_error("Invalid value for " + option_name + ": `" + value + "`.");
  }

  if (parsed_chars != value.size() || parsed_value <= 0) {
    throw std::runtime_error("Expected a positive integer for " + option_name + ".");
  }
  return parsed_value;
}

BenchmarkOptions parse_arguments(int argc, char** argv) {
  BenchmarkOptions options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto require_value = [&](const std::string& option_name) -> std::string {
      if (index + 1 >= argc) {
        throw std::runtime_error("Missing value for " + option_name + ".");
      }
      return argv[++index];
    };

    if (argument == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    }
    if (argument == "--frames") {
      options.frame_count = static_cast<std::size_t>(
          parse_positive_int(require_value(argument), argument));
    } else if (argument == "--width") {
      options.width = parse_positive_int(require_value(argument), argument);
    } else if (argument == "--height") {
      options.height = parse_positive_int(require_value(argument), argument);
    } else if (argument == "--jpeg-quality") {
      options.jpeg_quality = parse_positive_int(require_value(argument), argument);
      if (options.jpeg_quality > 100) {
        throw std::runtime_error("Expected --jpeg-quality to be in [1, 100].");
      }
    } else if (argument == "--redis-namespace") {
      options.redis_namespace = require_value(argument);
    } else if (argument == "--redis-host") {
      options.redis_host = require_value(argument);
    } else if (argument == "--redis-port") {
      options.redis_port = parse_positive_int(require_value(argument), argument);
    } else if (argument == "--redis-key") {
      options.redis_key = require_value(argument);
    } else if (argument == "--zmq-endpoint") {
      options.zmq_endpoint = require_value(argument);
    } else if (argument == "--zmq-timeout-ms") {
      options.zmq_timeout_ms = parse_positive_int(require_value(argument), argument);
    } else {
      throw std::runtime_error("Unknown argument: `" + argument + "`.");
    }
  }

  return options;
}

std::vector<std::string> build_jpeg_payloads(const BenchmarkOptions& options) {
  std::vector<std::string> payloads;
  payloads.reserve(options.frame_count);

  cv::Mat frame(options.height, options.width, CV_8UC3);
  std::vector<unsigned char> encoded_buffer;
  encoded_buffer.reserve(static_cast<std::size_t>(options.width) *
                         static_cast<std::size_t>(options.height) * 3);
  cv::RNG rng(123456);
  const std::vector<int> encode_params = {cv::IMWRITE_JPEG_QUALITY,
                                          options.jpeg_quality};

  for (std::size_t frame_index = 0; frame_index < options.frame_count;
       ++frame_index) {
    rng.fill(frame, cv::RNG::UNIFORM, 0, 256);
    const cv::Scalar overlay_color(frame_index % 255,
                                   (frame_index * 17) % 255,
                                   (frame_index * 31) % 255);
    cv::rectangle(frame, cv::Rect(0, 0, options.width / 4, options.height / 4),
                  overlay_color, cv::FILLED);

    if (!cv::imencode(".jpg", frame, encoded_buffer, encode_params)) {
      throw std::runtime_error("Failed to JPEG-encode benchmark frame payload.");
    }

    payloads.emplace_back(reinterpret_cast<const char*>(encoded_buffer.data()),
                          encoded_buffer.size());
  }

  return payloads;
}

PayloadStats summarize_payloads(const std::vector<std::string>& payloads) {
  PayloadStats stats;
  if (payloads.empty()) {
    return stats;
  }

  stats.min_bytes = std::numeric_limits<std::size_t>::max();
  for (const auto& payload : payloads) {
    const std::size_t payload_size = payload.size();
    stats.total_bytes += payload_size;
    stats.min_bytes = std::min(stats.min_bytes, payload_size);
    stats.max_bytes = std::max(stats.max_bytes, payload_size);
  }
  stats.average_bytes = static_cast<double>(stats.total_bytes) /
                        static_cast<double>(payloads.size());
  return stats;
}

double bytes_to_mebibytes(const std::size_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

void print_payload_summary(const BenchmarkOptions& options,
                           const PayloadStats& stats) {
  std::cout << "Prepared " << options.frame_count << " JPEG frames at "
            << options.width << "x" << options.height << ".\n";
  std::cout << std::fixed << std::setprecision(2)
            << "Payload sizes: avg " << stats.average_bytes << " B, min "
            << stats.min_bytes << " B, max " << stats.max_bytes << " B, total "
            << bytes_to_mebibytes(stats.total_bytes) << " MiB.\n";
}

BenchmarkResult benchmark_redis(const BenchmarkOptions& options,
                                const std::vector<std::string>& payloads) {
  SaiCommon::RedisClient redis_client(options.redis_namespace);
  redis_client.connect(options.redis_host, options.redis_port);

  const auto start_time = std::chrono::steady_clock::now();
  std::size_t total_bytes = 0;
  for (const auto& payload : payloads) {
    redis_client.set(options.redis_key, payload);
    total_bytes += payload.size();
  }
  const auto end_time = std::chrono::steady_clock::now();

  BenchmarkResult result;
  result.transport_name = "Redis SET";
  result.frame_count = payloads.size();
  result.total_bytes = total_bytes;
  result.elapsed_seconds =
      std::chrono::duration<double>(end_time - start_time).count();
  return result;
}

void throw_zmq_error(const std::string& action) {
  throw std::runtime_error(action + " failed: " + zmq_strerror(zmq_errno()));
}

void set_zmq_socket_option(void* socket,
                           const int option_name,
                           const int option_value,
                           const std::string& option_label) {
  if (zmq_setsockopt(socket, option_name, &option_value,
                     sizeof(option_value)) != 0) {
    throw_zmq_error("zmq_setsockopt(" + option_label + ")");
  }
}

BenchmarkResult benchmark_zmq(const BenchmarkOptions& options,
                              const std::vector<std::string>& payloads) {
  void* context = zmq_ctx_new();
  if (!context) {
    throw_zmq_error("zmq_ctx_new");
  }

  std::promise<void> server_ready_promise;
  auto server_ready_future = server_ready_promise.get_future();
  std::exception_ptr server_exception;
  void* request_socket = nullptr;
  std::size_t transmitted_bytes = 0;

  std::thread server_thread([&] {
    void* reply_socket = nullptr;
    try {
      reply_socket = zmq_socket(context, ZMQ_REP);
      if (!reply_socket) {
        throw_zmq_error("zmq_socket(ZMQ_REP)");
      }

      set_zmq_socket_option(reply_socket, ZMQ_LINGER, 0, "ZMQ_LINGER");
      set_zmq_socket_option(reply_socket, ZMQ_RCVTIMEO, options.zmq_timeout_ms,
                            "ZMQ_RCVTIMEO");
      set_zmq_socket_option(reply_socket, ZMQ_SNDTIMEO, options.zmq_timeout_ms,
                            "ZMQ_SNDTIMEO");

      if (zmq_bind(reply_socket, options.zmq_endpoint.c_str()) != 0) {
        throw_zmq_error("zmq_bind(" + options.zmq_endpoint + ")");
      }

      server_ready_promise.set_value();

      for (std::size_t frame_index = 0; frame_index < payloads.size();
           ++frame_index) {
        zmq_msg_t frame_message;
        if (zmq_msg_init(&frame_message) != 0) {
          throw_zmq_error("zmq_msg_init");
        }

        if (zmq_msg_recv(&frame_message, reply_socket, 0) == -1) {
          zmq_msg_close(&frame_message);
          throw_zmq_error("zmq_msg_recv");
        }

        if (zmq_msg_close(&frame_message) != 0) {
          throw_zmq_error("zmq_msg_close");
        }

        static constexpr char kAck[] = "1";
        if (zmq_send(reply_socket, kAck, sizeof(kAck) - 1, 0) == -1) {
          throw_zmq_error("zmq_send(ack)");
        }
      }
    } catch (...) {
      server_exception = std::current_exception();
      try {
        server_ready_promise.set_exception(std::current_exception());
      } catch (const std::future_error&) {
      }
    }

    if (reply_socket) {
      zmq_close(reply_socket);
    }
  });

  try {
    server_ready_future.get();

    request_socket = zmq_socket(context, ZMQ_REQ);
    if (!request_socket) {
      throw_zmq_error("zmq_socket(ZMQ_REQ)");
    }

    set_zmq_socket_option(request_socket, ZMQ_LINGER, 0, "ZMQ_LINGER");
    set_zmq_socket_option(request_socket, ZMQ_RCVTIMEO, options.zmq_timeout_ms,
                          "ZMQ_RCVTIMEO");
    set_zmq_socket_option(request_socket, ZMQ_SNDTIMEO, options.zmq_timeout_ms,
                          "ZMQ_SNDTIMEO");

    if (zmq_connect(request_socket, options.zmq_endpoint.c_str()) != 0) {
      throw_zmq_error("zmq_connect(" + options.zmq_endpoint + ")");
    }

    const auto start_time = std::chrono::steady_clock::now();
    for (const auto& payload : payloads) {
      if (zmq_send(request_socket, payload.data(), payload.size(), 0) == -1) {
        throw_zmq_error("zmq_send(frame)");
      }

      char ack[2] = {};
      if (zmq_recv(request_socket, ack, sizeof(ack), 0) == -1) {
        throw_zmq_error("zmq_recv(ack)");
      }

      transmitted_bytes += payload.size();
    }
    const auto end_time = std::chrono::steady_clock::now();

    if (request_socket) {
      zmq_close(request_socket);
      request_socket = nullptr;
    }

    if (server_thread.joinable()) {
      server_thread.join();
    }

    zmq_ctx_shutdown(context);
    zmq_ctx_term(context);

    if (server_exception) {
      std::rethrow_exception(server_exception);
    }

    BenchmarkResult result;
    result.transport_name = "ZMQ REQ/REP";
    result.frame_count = payloads.size();
    result.total_bytes = transmitted_bytes;
    result.elapsed_seconds =
        std::chrono::duration<double>(end_time - start_time).count();
    return result;
  } catch (...) {
    if (request_socket) {
      zmq_close(request_socket);
    }
    zmq_ctx_shutdown(context);
    if (server_thread.joinable()) {
      server_thread.join();
    }
    zmq_ctx_term(context);
    throw;
  }
}

void print_result(const BenchmarkResult& result) {
  const double elapsed_ms = result.elapsed_seconds * 1000.0;
  const double average_ms = elapsed_ms / static_cast<double>(result.frame_count);
  const double throughput_mib_s = bytes_to_mebibytes(result.total_bytes) /
                                  result.elapsed_seconds;

  std::cout << result.transport_name << ":\n"
            << "  Frames sent: " << result.frame_count << "\n"
            << std::fixed << std::setprecision(3)
            << "  Total time: " << elapsed_ms << " ms\n"
            << "  Avg/frame:  " << average_ms << " ms\n"
            << "  Throughput: " << throughput_mib_s << " MiB/s\n";
}

void print_comparison(const BenchmarkResult& redis_result,
                      const BenchmarkResult& zmq_result) {
  const BenchmarkResult* faster = &redis_result;
  const BenchmarkResult* slower = &zmq_result;
  if (zmq_result.elapsed_seconds < redis_result.elapsed_seconds) {
    faster = &zmq_result;
    slower = &redis_result;
  }

  const double ratio = slower->elapsed_seconds / faster->elapsed_seconds;
  std::cout << "Comparison:\n"
            << "  Faster path: " << faster->transport_name << "\n"
            << std::fixed << std::setprecision(3)
            << "  Speedup:     " << ratio << "x\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const BenchmarkOptions options = parse_arguments(argc, argv);
    const std::vector<std::string> payloads = build_jpeg_payloads(options);
    const PayloadStats payload_stats = summarize_payloads(payloads);

    print_payload_summary(options, payload_stats);
    std::cout << "Redis target: " << options.redis_host << ":"
              << options.redis_port << " / namespace `"
              << options.redis_namespace << "` / key `" << options.redis_key
              << "`\n";
    std::cout << "ZMQ endpoint: " << options.zmq_endpoint << "\n\n";

    const BenchmarkResult redis_result = benchmark_redis(options, payloads);
    const BenchmarkResult zmq_result = benchmark_zmq(options, payloads);

    print_result(redis_result);
    std::cout << '\n';
    print_result(zmq_result);
    std::cout << '\n';
    print_comparison(redis_result, zmq_result);
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "image_benchmark failed: " << exception.what() << "\n";
    return 1;
  }
}
 #include <opencv2/imgproc.hpp>
