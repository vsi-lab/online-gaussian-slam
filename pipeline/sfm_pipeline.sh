#!/usr/bin/env bash
# COLMAP (incremental) vs GLOMAP (global) poses on a converted TurtleBot dataset, then 3DGS on each.
# Same undistorted images and the same known PINHOLE intrinsics for both; shared feature extraction +
# sequential matching; only the mapper differs. Every stage is timed into <out>/timing.txt.
#
# Usage (WSL): bash sfm_pipeline.sh <dataset_dir> <out_dir> [3dgs_iters]
#   dataset_dir: output of bag_to_tum.py (rgb/, rgb.txt, pose.txt) ; config yaml beside it gives intrinsics
set -eo pipefail
DS=$(realpath "$1"); OUT=$(realpath -m "$2"); ITERS=${3:-30000}
CFG=$(ls ~/MonoGS/configs/rgbd/turtlebot/$(basename "$DS").yaml)
read FX FY CX CY W H < <(python3 - "$CFG" <<'EOF'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))["Dataset"]["Calibration"]
print(c["fx"], c["fy"], c["cx"], c["cy"], c["width"], c["height"])
EOF
)
mkdir -p "$OUT/images"
# images listed in rgb.txt (one sequence) -> plain copies so all tools see the same files
cut -d' ' -f2 "$DS/rgb.txt" | while read f; do [ -e "$OUT/images/$(basename $f)" ] || cp "$DS/$f" "$OUT/images/"; done
N=$(ls "$OUT/images" | wc -l)
source ~/miniconda3/etc/profile.d/conda.sh
T="$OUT/timing.txt"; touch "$T"
tick() { date +%s.%N; }
log() { grep -v "^$1 " "$T" > "$T.tmp" || true; mv "$T.tmp" "$T"; echo "$1 $(python3 -c "print(f'{$3-$2:.1f}')")" | tee -a "$T"; }
done_() { grep -q "^$1 " "$T"; }

conda activate colmap
DB="$OUT/database.db"
if ! done_ matching_s; then
rm -f "$DB"
t0=$(tick)
colmap feature_extractor --database_path "$DB" --image_path "$OUT/images" \
  --ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE \
  --ImageReader.camera_params "$FX,$FY,$CX,$CY" --FeatureExtraction.use_gpu 1 > "$OUT/log_extract.txt" 2>&1
t1=$(tick); log features_s $t0 $t1
colmap sequential_matcher --database_path "$DB" --SequentialMatching.overlap 15 \
  --FeatureMatching.use_gpu 1 > "$OUT/log_match.txt" 2>&1
t2=$(tick); log matching_s $t1 $t2
fi

# COLMAP incremental mapper (intrinsics fixed: they are known)
if ! done_ colmap_mapper_s; then
mkdir -p "$OUT/colmap/sparse"
cp "$DB" "$OUT/colmap/database.db"
t3=$(tick)
colmap mapper --database_path "$OUT/colmap/database.db" --image_path "$OUT/images" --output_path "$OUT/colmap/sparse" \
  --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0 \
  > "$OUT/log_colmap_mapper.txt" 2>&1
t4=$(tick); log colmap_mapper_s $t3 $t4
fi

# GLOMAP = COLMAP 4.2's global_mapper (GLOMAP merged into COLMAP). Standalone GLOMAP 1.2 cannot open a
# COLMAP 4.2 database (colmap::Database::Open throws), so the integrated version is used on the same database.
if ! done_ glomap_mapper_s; then
rm -rf "$OUT/glomap"; mkdir -p "$OUT/glomap/sparse"
cp "$DB" "$OUT/glomap/database.db"
t5=$(tick)
colmap global_mapper --database_path "$OUT/glomap/database.db" --image_path "$OUT/images" --output_path "$OUT/glomap/sparse"   > "$OUT/log_glomap_mapper.txt" 2>&1
t6=$(tick); log glomap_mapper_s $t5 $t6
fi
done_ images || echo "images $N" | tee -a "$T"

# biggest model of each -> sparse/0 for 3DGS; export text for pose eval
conda activate colmap
for M in colmap glomap; do
  best=$(for d in "$OUT/$M/sparse"/*/; do echo "$(colmap model_analyzer --path "$d" 2>&1 | sed -n 's/.*Registered images: *\([0-9]*\).*/\1/p') $d"; done | sort -n | tail -1)
  reg=${best%% *}; dir=${best#* }
  grep -v "^${M}_registered " "$T" > "$T.tmp" || true; mv "$T.tmp" "$T"; echo "${M}_registered $reg" | tee -a "$T"
  if [ ! -d "$OUT/gs_$M/model" ]; then
  rm -rf "$OUT/gs_$M"; mkdir -p "$OUT/gs_$M/sparse/0"
  ln -s "$OUT/images" "$OUT/gs_$M/images"
  cp "$dir"/*.bin "$OUT/gs_$M/sparse/0/"
  fi
  mkdir -p "$OUT/$M/txt"
  colmap model_converter --input_path "$dir" --output_path "$OUT/$M/txt" --output_type TXT > /dev/null 2>&1
done

# 3D Gaussian Splatting (graphdeco) on each pose set; --eval holds out every 8th image as test
conda activate gs3d
cd ~/gaussian-splatting
for M in colmap glomap; do
  if ! done_ 3dgs_train_${M}_s; then
  t7=$(tick)
  python train.py -s "$OUT/gs_$M" -m "$OUT/gs_$M/model" --eval --iterations $ITERS \
    --test_iterations $ITERS --save_iterations $ITERS --disable_viewer > "$OUT/log_3dgs_$M.txt" 2>&1
  t8=$(tick); log 3dgs_train_${M}_s $t7 $t8
  fi
  [ -s "$OUT/metrics_3dgs_$M.txt" ] && grep -q PSNR "$OUT/metrics_3dgs_$M.txt" && { grep -E "SSIM|PSNR|LPIPS" "$OUT/metrics_3dgs_$M.txt" | sed "s/^/  $M /"; continue; }
  python render.py -m "$OUT/gs_$M/model" --skip_train > /dev/null 2>&1
  python metrics.py -m "$OUT/gs_$M/model" > "$OUT/metrics_3dgs_$M.txt" 2>&1
  grep -E "SSIM|PSNR|LPIPS" "$OUT/metrics_3dgs_$M.txt" | sed "s/^/  $M /" || echo "  $M metrics FAILED (see metrics_3dgs_$M.txt)"
done
echo "done -> $OUT"
