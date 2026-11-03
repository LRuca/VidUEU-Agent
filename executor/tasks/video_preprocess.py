# executor/tasks/video_preprocess.py

import os
import cv2
import subprocess
from typing import List, Tuple, Dict, Any


class VideoPreprocessor:
    """
    T0: 视频预处理
    - 从输入视频中抽帧
    - 提取音频，保存为 16k mono wav

    修正版：
    - 抽帧时同时记录每一张帧对应的「真实时间戳」（单位：秒）
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        t0_cfg = config.get("tasks", {}).get("T0_video_preprocess", {})
        # 每秒抽多少帧，默认 1fps
        self.target_fps = t0_cfg.get("target_fps", 1)

    def extract_frames(self, video_path: str, frames_dir: str) -> Tuple[List[str], List[float]]:
        """
        抽帧，并同时返回：
        - frame_paths: 每一张保存的帧路径
        - frame_timestamps: 每一张帧在原视频中的时间戳（秒）

        说明（MIN-MOD + one more step + robust）：
        - 先用 OpenCV 抽帧（保持原逻辑）
        - 如果 OpenCV 抽到 0 帧，fallback 用 ffmpeg 抽帧
        - ⭐ 关键修复：ffmpeg 可能 exit=0 但产出 0 帧（典型 AV1 Missing Sequence Header），此时必须当作失败继续尝试
        - ⭐ NEW: ffmpeg 抽帧失败/产出0帧 -> remux(copy) 一次 -> 再抽帧
        - 如果仍失败：转码 H.264（libx264）-> 再抽帧
        """
        import glob
        import tempfile

        os.makedirs(frames_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if not orig_fps or orig_fps <= 0:
            orig_fps = 25.0

        target_fps = self.target_fps if self.target_fps and self.target_fps > 0 else 1.0
        step = max(int(orig_fps // target_fps), 1)

        frame_paths: List[str] = []
        frame_timestamps: List[float] = []

        idx = 0
        saved = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if idx % step == 0:
                sec = idx / orig_fps
                out_path = os.path.join(frames_dir, f"frame_{saved:04d}.jpg")
                cv2.imwrite(out_path, frame)
                frame_paths.append(out_path)
                frame_timestamps.append(sec)
                saved += 1

            idx += 1

        cap.release()

        def _cleanup_frames_dir():
            for fp in glob.glob(os.path.join(frames_dir, "frame_*.jpg")):
                try:
                    os.remove(fp)
                except Exception:
                    pass

        def _list_frames() -> List[str]:
            return sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))

        def _ffmpeg_extract(src_video: str) -> Tuple[List[str], List[float]]:
            """
            ffmpeg 抽帧：必须检查是否真的产出 jpg。
            两次尝试：
            1) 常规 fps 抽帧（hwaccel none）
            2) 若 0 帧，再试「强制软件解码 libdav1d + ignore_err + discardcorrupt + skip_frame nokey」
            """
            _cleanup_frames_dir()
            out_pat = os.path.join(frames_dir, "frame_%04d.jpg")

            # 1) normal extract
            cmd1 = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel", "fatal",  # 防刷屏：我们用“产出帧数”判断
                "-hwaccel", "none",
                "-analyzeduration", "200M",
                "-probesize", "200M",
                "-i", src_video,
                "-vf", f"fps={float(target_fps)}",
                "-q:v", "2",
                "-start_number", "0",
                out_pat,
            ]
            subprocess.run(cmd1, check=False)
            files = _list_frames()
            if len(files) > 0:
                ts = [i / float(target_fps) for i in range(len(files))]
                return files, ts

            # 2) retry: software decode + ignore errors + keyframes-only
            _cleanup_frames_dir()
            cmd2 = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel", "fatal",
                "-hwaccel", "none",
                "-c:v", "libdav1d",
                "-err_detect", "ignore_err",
                "-fflags", "+discardcorrupt",
                "-analyzeduration", "200M",
                "-probesize", "200M",
                "-skip_frame", "nokey",
                "-i", src_video,
                "-vf", f"fps={float(target_fps)}",
                "-vsync", "0",
                "-q:v", "2",
                "-start_number", "0",
                out_pat,
            ]
            subprocess.run(cmd2, check=False)
            files = _list_frames()
            if len(files) > 0:
                ts = [i / float(target_fps) for i in range(len(files))]
                return files, ts

            raise RuntimeError("ffmpeg_extract_produced_0_frames")

        def _ffmpeg_remux_copy(src_video: str) -> str:
            """
            多一步：先 remux(copy) 生成一个新的 mp4（不解码，低成本）
            有时能修复容器/track extradata 类问题。
            """
            fd, tmp_path = tempfile.mkstemp(prefix="remux_", suffix=".mp4", dir=frames_dir)
            os.close(fd)
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel", "fatal",
                "-hwaccel", "none",
                "-err_detect", "ignore_err",
                "-fflags", "+discardcorrupt",
                "-analyzeduration", "200M",
                "-probesize", "200M",
                "-i", src_video,
                "-map", "0:v:0",
                "-c", "copy",
                "-movflags", "+faststart",
                tmp_path,
            ]
            subprocess.run(cmd, check=False)
            # 即使 remux exit=0，也要保证文件存在且非空
            if (not os.path.exists(tmp_path)) or (os.path.getsize(tmp_path) <= 0):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                raise RuntimeError("ffmpeg_remux_failed")
            return tmp_path

        def _ffmpeg_transcode_to_h264(src_video: str) -> str:
            # 生成临时 h264 mp4（需要解码，若源片段真坏可能仍失败）
            fd, tmp_path = tempfile.mkstemp(prefix="transcoded_", suffix=".mp4", dir=frames_dir)
            os.close(fd)
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel", "fatal",
                "-hwaccel", "none",
                "-err_detect", "ignore_err",
                "-fflags", "+discardcorrupt",
                "-analyzeduration", "200M",
                "-probesize", "200M",
                "-i", src_video,
                "-an",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "veryfast",
                "-crf", "23",
                tmp_path,
            ]
            subprocess.run(cmd, check=False)
            if (not os.path.exists(tmp_path)) or (os.path.getsize(tmp_path) <= 0):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                raise RuntimeError("ffmpeg_transcode_failed")
            return tmp_path

        # ===================== ✅ fallback for AV1 / decode failures =====================
        if len(frame_paths) == 0:
            # 1) try ffmpeg extract (must produce >0 frames, else raises)
            try:
                files, ts = _ffmpeg_extract(video_path)
                frame_paths, frame_timestamps = files, ts
                print(f"⚠️  [T0] OpenCV 抽帧为 0，已 fallback 到 ffmpeg 抽帧，frames={len(frame_paths)}")
            except Exception as e1:
                print(f"⚠️  [T0 WARN] ffmpeg 抽帧失败/产出0帧，将尝试 remux 再抽帧: {e1}")

                tmp_remux = ""
                try:
                    # 1.5) remux then extract
                    tmp_remux = _ffmpeg_remux_copy(video_path)
                    files, ts = _ffmpeg_extract(tmp_remux)
                    frame_paths, frame_timestamps = files, ts
                    print(f"⚠️  [T0] remux 后抽帧成功，frames={len(frame_paths)}")
                except Exception as e_remux:
                    print(f"⚠️  [T0 WARN] remux 后仍失败/产出0帧，将尝试转码 H.264 再抽帧: {e_remux}")

                    # 2) transcode then extract
                    tmp_h264 = ""
                    try:
                        tmp_h264 = _ffmpeg_transcode_to_h264(video_path)
                        files, ts = _ffmpeg_extract(tmp_h264)
                        frame_paths, frame_timestamps = files, ts
                        print(f"⚠️  [T0] 已转码 H.264 后抽帧成功，frames={len(frame_paths)}")
                    except Exception as e2:
                        print(f"⚠️  [T0] 转码 H.264 + 抽帧仍失败/产出0帧: {e2}")
                    finally:
                        if tmp_h264 and os.path.exists(tmp_h264):
                            try:
                                os.remove(tmp_h264)
                            except Exception:
                                pass
                finally:
                    if tmp_remux and os.path.exists(tmp_remux):
                        try:
                            os.remove(tmp_remux)
                        except Exception:
                            pass

        print(
            f"✅ 已提取 {len(frame_paths)} 帧 到 {frames_dir} | "
            f"orig_fps={orig_fps:.2f}, target_fps={target_fps}, step={step}"
        )
        return frame_paths, frame_timestamps

    def extract_audio(self, video_path: str, audio_path: str) -> str:
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)

        # 第一次尝试：正常从视频中抽取音频
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            audio_path,
        ]
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except subprocess.CalledProcessError:
            # 可能是视频本身没有音频轨道，我们兜底生成一段静音音频
            print("⚠️  视频可能无音频轨道，改为生成静音 audio.wav")
            silent_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=16000:cl=mono",
                "-t",
                "4",  # 生成 4 秒静音，可按需调整
                audio_path,
            ]
            subprocess.run(
                silent_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

        if not os.path.exists(audio_path):
            raise RuntimeError(f"音频提取失败，未找到文件: {audio_path}")

        print(f"✅ 音频提取完成: {audio_path}")
        return audio_path

    def run(self, video_path: str, temp_dir: str = None) -> Dict[str, Any]:
        """
        返回:
            {
              "video_path": <原始或切分后的视频路径>,
              "frames": List[str],              # 抽帧路径
              "frame_timestamps": List[float],  # 每张帧对应的时间戳（秒）
              "audio_path": str                 # 音频 wav 路径
            }
        """
        # 如果外部没传 temp_dir，就从 config 或默认路径中取
        if temp_dir is None:
            paths_cfg = self.config.get("paths", {})
            temp_dir = paths_cfg.get("temp_dir", "data/temp")

        os.makedirs(temp_dir, exist_ok=True)

        frames_dir = os.path.join(temp_dir, "frames")
        audio_path = os.path.join(temp_dir, "audio.wav")

        print("🔧 T0: 视频预处理阶段开始")
        frame_paths, frame_timestamps = self.extract_frames(video_path, frames_dir)
        audio = self.extract_audio(video_path, audio_path)

        return {
            "video_path": video_path,
            "frames": frame_paths,
            "frame_timestamps": frame_timestamps,
            "audio_path": audio,
        }


def extract_frames_with_timestamps(
    video_path: str,
    fps: float = 1.5,
    out_dir: str = None,
) -> List[Dict[str, Any]]:
    """
    抽帧并返回 [{'img': 路径, 'sec': 时间戳}]
    ⭐ MIN-MOD + robust:
    - OpenCV 抽到 0 帧 -> ffmpeg 抽帧（必须产出 >0 帧，否则继续尝试）
    - ⭐ NEW: ffmpeg 产出0帧/失败 -> remux(copy) -> 再抽帧
    - 仍失败 -> 转码 H.264 -> 再抽帧
    """
    import glob
    import tempfile
    from typing import Any, Dict, List

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fps = float(fps) if fps and float(fps) > 0 else 1.0
    interval = max(int(float(orig_fps) / float(fps)), 1)

    basename = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = out_dir if out_dir is not None else f"/tmp/t2_frames/{basename}"
    os.makedirs(save_dir, exist_ok=True)

    frame_list: List[Dict[str, Any]] = []
    idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            sec = idx / float(orig_fps)
            out_path = os.path.join(save_dir, f"frame_{saved:04d}.jpg")
            cv2.imwrite(out_path, frame)
            frame_list.append({"img": out_path, "sec": round(sec, 2)})
            saved += 1
        idx += 1

    cap.release()

    def _cleanup_frames():
        for fp in glob.glob(os.path.join(save_dir, "frame_*.jpg")):
            try:
                os.remove(fp)
            except Exception:
                pass

    def _list_frames() -> List[str]:
        return sorted(glob.glob(os.path.join(save_dir, "frame_*.jpg")))

    def _ffmpeg_extract(src_video: str) -> List[Dict[str, Any]]:
        """
        两次尝试：
        1) 常规 fps 抽帧（hwaccel none）
        2) 若 0 帧，再试「libdav1d + ignore_err + discardcorrupt + skip_frame nokey」
        产出 0 帧 => raise（防止 ffmpeg exit=0 假成功）。
        """
        _cleanup_frames()
        out_pat = os.path.join(save_dir, "frame_%04d.jpg")

        cmd1 = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel", "fatal",
            "-hwaccel", "none",
            "-analyzeduration", "200M",
            "-probesize", "200M",
            "-i", src_video,
            "-vf", f"fps={float(fps)}",
            "-q:v", "2",
            "-start_number", "0",
            out_pat,
        ]
        subprocess.run(cmd1, check=False)
        files = _list_frames()
        if len(files) > 0:
            return [{"img": p, "sec": round(i / float(fps), 2)} for i, p in enumerate(files)]

        _cleanup_frames()
        cmd2 = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel", "fatal",
            "-hwaccel", "none",
            "-c:v", "libdav1d",
            "-err_detect", "ignore_err",
            "-fflags", "+discardcorrupt",
            "-analyzeduration", "200M",
            "-probesize", "200M",
            "-skip_frame", "nokey",
            "-i", src_video,
            "-vf", f"fps={float(fps)}",
            "-vsync", "0",
            "-q:v", "2",
            "-start_number", "0",
            out_pat,
        ]
        subprocess.run(cmd2, check=False)
        files = _list_frames()
        if len(files) > 0:
            return [{"img": p, "sec": round(i / float(fps), 2)} for i, p in enumerate(files)]

        raise RuntimeError("ffmpeg_extract_produced_0_frames")

    def _ffmpeg_remux_copy(src_video: str) -> str:
        fd, tmp_path = tempfile.mkstemp(prefix="remux_", suffix=".mp4", dir=save_dir)
        os.close(fd)
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel", "fatal",
            "-hwaccel", "none",
            "-err_detect", "ignore_err",
            "-fflags", "+discardcorrupt",
            "-analyzeduration", "200M",
            "-probesize", "200M",
            "-i", src_video,
            "-map", "0:v:0",
            "-c", "copy",
            "-movflags", "+faststart",
            tmp_path,
        ]
        subprocess.run(cmd, check=False)
        if (not os.path.exists(tmp_path)) or (os.path.getsize(tmp_path) <= 0):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            raise RuntimeError("ffmpeg_remux_failed")
        return tmp_path

    def _ffmpeg_transcode_to_h264(src_video: str) -> str:
        fd, tmp_path = tempfile.mkstemp(prefix="transcoded_", suffix=".mp4", dir=save_dir)
        os.close(fd)
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel", "fatal",
            "-hwaccel", "none",
            "-err_detect", "ignore_err",
            "-fflags", "+discardcorrupt",
            "-analyzeduration", "200M",
            "-probesize", "200M",
            "-i", src_video,
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-crf", "23",
            tmp_path,
        ]
        subprocess.run(cmd, check=False)
        if (not os.path.exists(tmp_path)) or (os.path.getsize(tmp_path) <= 0):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            raise RuntimeError("ffmpeg_transcode_failed")
        return tmp_path

    # ===================== ✅ fallback for AV1 / decode failures =====================
    if len(frame_list) == 0:
        try:
            frame_list = _ffmpeg_extract(video_path)
            print(f"⚠️  [T2] OpenCV 抽帧为 0，已 fallback 到 ffmpeg 抽帧，frames={len(frame_list)}")
        except Exception as e1:
            print(f"⚠️  [T2 WARN] ffmpeg 抽帧失败/产出0帧，将尝试 remux 再抽帧: {e1}")

            tmp_remux = ""
            try:
                tmp_remux = _ffmpeg_remux_copy(video_path)
                frame_list = _ffmpeg_extract(tmp_remux)
                print(f"⚠️  [T2] remux 后抽帧成功，frames={len(frame_list)}")
            except Exception as e_remux:
                print(f"⚠️  [T2 WARN] remux 后仍失败/产出0帧，将尝试转码 H.264 再抽帧: {e_remux}")

                tmp_h264 = ""
                try:
                    tmp_h264 = _ffmpeg_transcode_to_h264(video_path)
                    frame_list = _ffmpeg_extract(tmp_h264)
                    print(f"⚠️  [T2] 已转码 H.264 后抽帧成功，frames={len(frame_list)}")
                except Exception as e2:
                    print(f"⚠️  [T2] 转码 H.264 + 抽帧仍失败/产出0帧: {e2}")
                    return []
                finally:
                    if tmp_h264 and os.path.exists(tmp_h264):
                        try:
                            os.remove(tmp_h264)
                        except Exception:
                            pass
            finally:
                if tmp_remux and os.path.exists(tmp_remux):
                    try:
                        os.remove(tmp_remux)
                    except Exception:
                        pass

    return frame_list
