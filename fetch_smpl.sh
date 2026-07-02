#!/bin/bash
# Downloads the license-gated SMPL body models into dataset/body_models/smpl/.
# Run this yourself, inside WSL, from the WHAM checkout — it needs YOUR OWN
# credentials for smpl.is.tue.mpg.de and smplify.is.tue.mpg.de (same account
# usually works for both; register first if you haven't).
#
# Usage (from inside /root/WHAM):
#   bash fetch_smpl.sh
set -e
urle () { local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }
cd "$(dirname "$0")"
mkdir -p dataset/body_models/smpl

echo "Register first at https://smplify.is.tue.mpg.de and https://smpl.is.tue.mpg.de"
echo "if you haven't already (same account usually works for both)."
read -p "MPI username (email): " USER_IN
echo "MPI password (input is hidden — just type it and press Enter):"
read -sp "> " PASS_IN
echo
U=$(urle "$USER_IN"); P=$(urle "$PASS_IN")

check_zip () {
    # A failed/unauthorized download returns a small HTML error page, not a
    # zip. Catch that early instead of confusing "not a zip file" errors later.
    local f="$1" site="$2"
    local size
    size=$(stat -c%s "$f" 2>/dev/null || echo 0)
    if [ "$size" -lt 100000 ]; then
        echo "ERROR: download from $site looks wrong (only $size bytes — likely"
        echo "a login/permission error page, not the zip). Common causes:"
        echo "  - wrong username/password"
        echo "  - you registered but haven't accepted that site's license yet"
        echo "    (log into $site in a browser and check for a pending agreement)"
        exit 1
    fi
}

echo "-- SMPL neutral (from SMPLify) --"
wget --post-data "username=$U&password=$P" 'https://download.is.tue.mpg.de/download.php?domain=smplify&resume=1&sfile=mpips_smplify_public_v2.zip' -O dataset/body_models/smplify.zip --no-check-certificate --continue
check_zip dataset/body_models/smplify.zip https://smplify.is.tue.mpg.de
unzip -o dataset/body_models/smplify.zip -d dataset/body_models/smplify
mv dataset/body_models/smplify/smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl dataset/body_models/smpl/SMPL_NEUTRAL.pkl
rm -rf dataset/body_models/smplify dataset/body_models/smplify.zip

echo "-- SMPL male/female (from SMPL) --"
wget --post-data "username=$U&password=$P" 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.0.0.zip' -O dataset/body_models/smpl.zip --no-check-certificate --continue
check_zip dataset/body_models/smpl.zip https://smpl.is.tue.mpg.de
unzip -o dataset/body_models/smpl.zip -d dataset/body_models/smpl_tmp
mv dataset/body_models/smpl_tmp/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl dataset/body_models/smpl/SMPL_FEMALE.pkl
mv dataset/body_models/smpl_tmp/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl dataset/body_models/smpl/SMPL_MALE.pkl
rm -rf dataset/body_models/smpl_tmp dataset/body_models/smpl.zip

echo ""
echo "Done. SMPL files:"
ls -la dataset/body_models/smpl/
echo ""
echo "WHAM is now fully set up. In the Bonezzzz app, select the WHAM backend"
echo "on the Pose Estimation node and press Run."
