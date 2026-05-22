from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from extractor import extract_to_parquet as mod


class ExtractToParquetTests(unittest.TestCase):
    def test_build_prune_keep_mask_reuses_edge_trim_and_stationary_pruning(self) -> None:
        timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        keep_mask = mod.build_prune_keep_mask(
            timestamps=timestamps,
            positions=positions,
            trim_start=1,
            trim_end=0,
            vel_thresh=1e-6,
            stationary_window=2,
        )

        self.assertEqual(keep_mask.tolist(), [False, True, True, False, False])

    def test_project_tracks_to_world_points_uses_stationary_camera_basis(self) -> None:
        cv2 = mod._require_cv2()

        with tempfile.TemporaryDirectory() as tmp_dir:
            depth_path = Path(tmp_dir) / "frame_000000.png"
            depth_frame_mm = np.full((4, 4), 1000, dtype=np.uint16)
            self.assertTrue(cv2.imwrite(str(depth_path), depth_frame_mm))

            tracks = np.array([[[1.0, 1.0], [2.0, 2.0]]], dtype=np.float32)
            visibility = np.array([[True, False]], dtype=bool)
            source_frame_indices = np.array([0], dtype=np.int64)
            calibration = mod.CameraCalibration(
                fovy_degrees=60.0,
                camera_position_world=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                camera_right_world=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                camera_down_world=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                camera_forward_world=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            )

            projected = mod.project_tracks_to_world_points(
                tracks=tracks,
                visibility=visibility,
                depth_frame_paths=[depth_path],
                source_frame_indices=source_frame_indices,
                camera_calibration=calibration,
                expected_frame_shape=(4, 4),
            )

        fx, fy, cx, cy = mod.compute_camera_intrinsics(4, 4, 60.0)
        expected_point = np.array(
            [
                (1.0 - cx) / fx,
                (1.0 - cy) / fy,
                1.0,
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(projected[0, 0], expected_point, atol=1e-5)
        self.assertTrue(np.isnan(projected[0, 1]).all())

    def test_create_depth_point_clouds_writes_only_chunked_outputs(self) -> None:
        cv2 = mod._require_cv2()

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            episode_dir = temp_root / "episode_000001"
            depth_dir = episode_dir / "visual" / "depth" / "depth_frames"
            depth_dir.mkdir(parents=True, exist_ok=True)

            frame_height = 8
            frame_width = 8
            aligned_frames = np.zeros((5, frame_height, frame_width, 3), dtype=np.uint8)
            for frame_index in range(len(aligned_frames)):
                depth_frame_mm = np.full((frame_height, frame_width), 1000 + frame_index, dtype=np.uint16)
                self.assertTrue(
                    cv2.imwrite(str(depth_dir / f"frame_{frame_index:06d}.png"), depth_frame_mm)
                )

            aligned_episode = mod.AlignedEpisode(
                source_dir=episode_dir,
                source_name="episode_000001",
                timestamps=np.arange(len(aligned_frames), dtype=np.float64),
                positions=np.zeros((len(aligned_frames), 3), dtype=np.float32),
                orientations=np.repeat(np.eye(3, dtype=np.float32)[None], len(aligned_frames), axis=0),
                frames=aligned_frames,
                source_frame_indices=np.arange(len(aligned_frames), dtype=np.int64),
                video_fps=30.0,
            )
            prune_keep_mask = np.array([True, False, True, True, False], dtype=bool)
            processed_episode = mod.ProcessedEpisode(
                source_dir=episode_dir,
                source_name="episode_000001",
                timestamps=aligned_episode.timestamps[prune_keep_mask],
                positions=aligned_episode.positions[prune_keep_mask],
                orientations=aligned_episode.orientations[prune_keep_mask],
                frames=aligned_episode.frames[prune_keep_mask],
                source_frame_indices=aligned_episode.source_frame_indices[prune_keep_mask],
                video_fps=aligned_episode.video_fps,
            )
            processing_result = mod.EpisodeProcessingResult(
                aligned_episode=aligned_episode,
                prune_keep_mask=prune_keep_mask,
                processed_episode=processed_episode,
            )

            def fake_track(cotracker_context, frames, sampled_pixels):
                tracks = np.broadcast_to(
                    sampled_pixels.astype(np.float32)[None, :, :],
                    (len(frames), len(sampled_pixels), 2),
                ).copy()
                visibility = np.ones((len(frames), len(sampled_pixels)), dtype=bool)
                return tracks, visibility

            sampled_pixels = np.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.int32)

            def fake_select_points(aligned_episode, camera_calibration, contact_spec):
                self.assertIs(aligned_episode, processing_result.aligned_episode)
                self.assertEqual(contact_spec.radius_m, 0.1)
                return sampled_pixels

            output_dir = temp_root / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            with mock.patch.object(mod, "track_sampled_pixels", fake_track), mock.patch.object(
                mod,
                "select_points_to_track",
                fake_select_points,
            ):
                wrote = mod.create_depth_point_clouds(
                    output_dir=output_dir,
                    output_episode_index=1,
                    processing_result=processing_result,
                    chunk_size=2,
                    cotracker_context=None,
                    camera_calibration=mod.CameraCalibration(
                        fovy_degrees=60.0,
                        camera_position_world=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                        camera_right_world=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                        camera_down_world=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                        camera_forward_world=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                    ),
                    contact_spec=mod.ContactCylinderSpec(radius_m=0.1, half_height_m=0.2),
                )

            self.assertTrue(wrote)
            pointcloud_episode_dir = output_dir / "point_clouds" / "episode_0001"
            self.assertTrue((pointcloud_episode_dir / "sampled_pixels.npy").exists())
            self.assertFalse((pointcloud_episode_dir / "point_clouds.npy").exists())
            np.testing.assert_array_equal(np.load(pointcloud_episode_dir / "sampled_pixels.npy"), sampled_pixels)

            chunk_paths = sorted(pointcloud_episode_dir.glob("chunk_*.npy"))
            self.assertEqual([path.name for path in chunk_paths], ["chunk_0001.npy", "chunk_0002.npy"])

            first_chunk = np.load(chunk_paths[0])
            second_chunk = np.load(chunk_paths[1])
            self.assertEqual(first_chunk.shape, (2, 4, 3))
            self.assertEqual(second_chunk.shape, (1, 4, 3))
            self.assertTrue(np.isfinite(first_chunk[..., 2]).all())

    def test_create_depth_point_clouds_skips_when_no_pixels_are_selected(self) -> None:
        cv2 = mod._require_cv2()

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            episode_dir = temp_root / "episode_000001"
            depth_dir = episode_dir / "visual" / "depth" / "depth_frames"
            depth_dir.mkdir(parents=True, exist_ok=True)
            depth_frame_mm = np.full((8, 8), 1000, dtype=np.uint16)
            self.assertTrue(cv2.imwrite(str(depth_dir / "frame_000000.png"), depth_frame_mm))

            aligned_episode = mod.AlignedEpisode(
                source_dir=episode_dir,
                source_name="episode_000001",
                timestamps=np.array([0.0], dtype=np.float64),
                positions=np.zeros((1, 3), dtype=np.float32),
                orientations=np.repeat(np.eye(3, dtype=np.float32)[None], 1, axis=0),
                frames=np.zeros((1, 8, 8, 3), dtype=np.uint8),
                source_frame_indices=np.array([0], dtype=np.int64),
                video_fps=30.0,
            )
            processing_result = mod.EpisodeProcessingResult(
                aligned_episode=aligned_episode,
                prune_keep_mask=np.array([True], dtype=bool),
                processed_episode=mod.ProcessedEpisode(
                    source_dir=aligned_episode.source_dir,
                    source_name=aligned_episode.source_name,
                    timestamps=aligned_episode.timestamps.copy(),
                    positions=aligned_episode.positions.copy(),
                    orientations=aligned_episode.orientations.copy(),
                    frames=aligned_episode.frames.copy(),
                    source_frame_indices=aligned_episode.source_frame_indices.copy(),
                    video_fps=aligned_episode.video_fps,
                ),
            )

            output_dir = temp_root / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            with mock.patch.object(
                mod,
                "select_points_to_track",
                return_value=np.zeros((0, 2), dtype=np.int32),
            ), mock.patch.object(mod, "track_sampled_pixels") as track_mock:
                wrote = mod.create_depth_point_clouds(
                    output_dir=output_dir,
                    output_episode_index=1,
                    processing_result=processing_result,
                    chunk_size=2,
                    cotracker_context=None,
                    camera_calibration=mod.CameraCalibration(
                        fovy_degrees=60.0,
                        camera_position_world=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                        camera_right_world=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                        camera_down_world=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                        camera_forward_world=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                    ),
                    contact_spec=mod.ContactCylinderSpec(radius_m=0.1, half_height_m=0.2),
                )

            self.assertFalse(wrote)
            track_mock.assert_not_called()
            self.assertFalse((output_dir / "point_clouds").exists())


if __name__ == "__main__":
    unittest.main()
