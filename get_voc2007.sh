#!/usr/bin/env bash
#
# Set up the PASCAL VOC 2007 dataset for the deepcluster VOC classification eval
# (deepcluster/eval_voc_classif.py).
#
# Usage:
#   ./get_voc2007.sh                        # into ./data next to this script
#   ./get_voc2007.sh --dest /scratch/data   # anywhere else (e.g. on the server)
#   ./get_voc2007.sh --keep-tars            # keep the archives after extracting
#   ./get_voc2007.sh --force                # re-extract even if already laid out
#
# Needs only curl or wget, tar, and a sha256 tool. No sudo, no python.
# Idempotent and resumable: re-running checks what is already on disk and
# fetches only what is missing or corrupt, so an interrupted download
# continues where it stopped rather than starting over.

set -euo pipefail

BASE_URL="http://host.robots.ox.ac.uk/pascal/VOC/voc2007"

# Archives to fetch, with the SHA-256 of each (verified against the Oxford host;
# the MD5s of these same files are the ones torchvision ships for VOC2007).
TARS=(
  "VOCtrainval_06-Nov-2007.tar"
  "VOCtest_06-Nov-2007.tar"
)
SHA256S=(
  "7d8cd951101b0957ddfd7a530bdc8a94f06121cfc1e511bb5937e973020c7508"
  "6836888e2e01dca84577a849d339fa4f73e1e4f135d312430c4856b5609b4892"
)

# What a complete VOC2007 tree contains once both archives are unpacked.
EXPECT_IMAGES=9963        # 5011 trainval + 4952 test
EXPECT_SPLIT_FILES=20     # one per class; eval_voc_classif.py asserts on this

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${SCRIPT_DIR}/data"
KEEP_TARS=0
FORCE=0

log()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
  sed -n '3,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dest)      DEST="${2:?--dest needs a path}"; shift 2 ;;
    --dest=*)    DEST="${1#*=}"; shift ;;
    --keep-tars) KEEP_TARS=1; shift ;;
    --force)     FORCE=1; shift ;;
    -h|--help)   usage ;;
    *)           die "unknown option: $1 (try --help)" ;;
  esac
done

# ---------------------------------------------------------------- helpers ---

have() { command -v "$1" >/dev/null 2>&1; }

if have curl;   then FETCHER=curl
elif have wget; then FETCHER=wget
else die "need curl or wget on PATH"; fi

if have sha256sum;  then sha_of() { sha256sum "$1" | cut -d' ' -f1; }
elif have shasum;   then sha_of() { shasum -a 256 "$1" | cut -d' ' -f1; }
else warn "no sha256sum/shasum found -- skipping integrity checks"
     sha_of() { echo "SKIP"; }
fi

# Resumes a partial file rather than restarting it.
fetch() {
  local url="$1" out="$2"
  # One-line progress; curl's default table is unreadable in a log file.
  if [ "$FETCHER" = curl ]; then
    curl -fL --retry 3 --retry-delay 2 -C - --progress-bar -o "$out" "$url"
  else
    wget --tries=3 --continue -q --show-progress -O "$out" "$url"
  fi
}

verify() {
  local file="$1" want="$2" got
  got="$(sha_of "$file")"
  [ "$got" = "SKIP" ] && return 0
  [ "$got" = "$want" ]
}

count_glob() {  # count matches without tripping `set -e` on no-match
  local n=0 f
  for f in $1; do [ -e "$f" ] && n=$((n + 1)); done
  echo "$n"
}

# ------------------------------------------------------------------- main ---

VOCROOT="${DEST}/VOCdevkit/VOC2007"

if [ "$FORCE" -eq 0 ] && [ -d "${VOCROOT}/JPEGImages" ]; then
  n_img=$(count_glob "${VOCROOT}/JPEGImages/*.jpg")
  if [ "$n_img" -eq "$EXPECT_IMAGES" ]; then
    log "VOC2007 already complete at ${VOCROOT} (${n_img} images) -- nothing to do."
    echo
    echo "  export VOCDIR=\"${VOCROOT}\""
    exit 0
  fi
  warn "found ${VOCROOT} with ${n_img}/${EXPECT_IMAGES} images -- completing it"
fi

mkdir -p "$DEST"
cd "$DEST"

log "destination: ${DEST}"

for i in "${!TARS[@]}"; do
  tar_name="${TARS[$i]}"
  want_sha="${SHA256S[$i]}"

  if [ -f "$tar_name" ] && verify "$tar_name" "$want_sha"; then
    log "${tar_name}: already present and verified"
  else
    if [ -f "$tar_name" ]; then
      warn "${tar_name}: incomplete or corrupt -- resuming download"
    fi
    log "downloading ${tar_name} ..."
    # A resume can't heal a file that is the right length but wrong content: the
    # server answers 416 and curl errors out. So treat a failed resume and a
    # still-bad checksum the same way -- drop the file and fetch it clean.
    if ! fetch "${BASE_URL}/${tar_name}" "$tar_name" || ! verify "$tar_name" "$want_sha"; then
      warn "${tar_name}: resume failed or checksum still bad -- refetching from scratch"
      rm -f "$tar_name"
      fetch "${BASE_URL}/${tar_name}" "$tar_name" || die "download failed for ${tar_name}"
      verify "$tar_name" "$want_sha" || die "${tar_name}: checksum mismatch, refusing to extract"
    fi
    log "${tar_name}: verified"
  fi
done

for tar_name in "${TARS[@]}"; do
  log "extracting ${tar_name} ..."
  tar -xf "$tar_name" -C "$DEST"
done

# ---------------------------------------------------------------- validate ---

log "validating layout ..."

[ -d "$VOCROOT" ] || die "expected ${VOCROOT} after extraction, not found"

n_img=$(count_glob "${VOCROOT}/JPEGImages/*.jpg")
n_ann=$(count_glob "${VOCROOT}/Annotations/*.xml")
n_trainval=$(count_glob "${VOCROOT}/ImageSets/Main/*_trainval.txt")
n_test=$(count_glob "${VOCROOT}/ImageSets/Main/*_test.txt")

printf '    JPEGImages              %5d  (expected %d)\n' "$n_img" "$EXPECT_IMAGES"
printf '    Annotations             %5d  (expected %d)\n' "$n_ann" "$EXPECT_IMAGES"
printf '    ImageSets/Main *_trainval  %2d  (expected %d)\n' "$n_trainval" "$EXPECT_SPLIT_FILES"
printf '    ImageSets/Main *_test      %2d  (expected %d)\n' "$n_test" "$EXPECT_SPLIT_FILES"

fail=0
[ "$n_img"      -eq "$EXPECT_IMAGES" ]      || { warn "wrong image count";      fail=1; }
[ "$n_ann"      -eq "$EXPECT_IMAGES" ]      || { warn "wrong annotation count"; fail=1; }
[ "$n_trainval" -eq "$EXPECT_SPLIT_FILES" ] || { warn "wrong trainval split count"; fail=1; }
[ "$n_test"     -eq "$EXPECT_SPLIT_FILES" ] || { warn "wrong test split count";     fail=1; }
[ "$fail" -eq 0 ] || die "layout validation failed -- re-run with --force"

if [ "$KEEP_TARS" -eq 0 ]; then
  log "removing archives (pass --keep-tars to keep them)"
  for tar_name in "${TARS[@]}"; do rm -f "${DEST}/${tar_name}"; done
fi

log "VOC 2007 ready."
echo
echo "  export VOCDIR=\"${VOCROOT}\""
echo
echo "then, from the deepcluster directory:"
echo "  VOCDIR=\"${VOCROOT}\" ./eval_voc_classif.sh"
