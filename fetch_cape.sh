#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# username and password input
echo -e "\nYou need to register CAPE via https://icon.is.tue.mpg.de/"
read -p "Username (CAPE):" username
read -p "Password (CAPE):" password
username=$(urle $username)
password=$(urle $password)

# CAPE downloading
echo -e "\nDownloading CAPE testset for ICON..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=cape&resume=1&sfile=icon_test.zip' -O './data/icon_test.zip' --no-check-certificate --continue
unzip ./data/icon_test.zip -d ./data/
rm ./data/icon_test.zip