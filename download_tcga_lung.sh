#!/bin/bash

# Get the base path where this script is located
BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

download_gdc_client() {
  echo "downloading gdc-client"
  # Ubuntu
  wget https://gdc.cancer.gov/files/public/file/gdc-client_v1.6.1_Ubuntu_x64.zip
  unzip gdc-client_v1.6.1_Ubuntu_x64.zip
  chmod +x gdc-client
}

download_diagnostic() {
  local manifest_path=$1
  local output_dir=$2
  local manifest_name=$(basename "$manifest_path")

  mkdir -p "$output_dir"

  cp "$manifest_path" "$output_dir"
  cd "$output_dir" || exit

  "$BASE_PATH"/gdc-client download -m "$manifest_name"
  cd "$BASE_PATH" || exit
  echo "Done"
}

if [[ ! -f "$BASE_PATH/gdc-client" ]]; then
  download_gdc_client
fi

download_diagnostic "datasets/tcga/manifests/0_luad/gdc_manifest_20230520_101102.txt" "datasets/tcga/0_luad"
download_diagnostic "datasets/tcga/manifests/1_lusc/gdc_manifest_20230520_101010.txt" "datasets/tcga/1_lusc"

total_files=$(($(ls datasets/tcga/lusc | wc -l) + $(ls datasets/tcga/luad | wc -l)))

echo "Total files downloaded: $total_files"

if ((total_files > 1030)); then
  echo "*** SUCCESSFUL ***"
else
  echo "*** ERROR RUN ONE MORE TIME ***"
fi
