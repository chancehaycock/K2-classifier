#!/bin/bash

# Choose Campaign Number to Check
echo "Enter campaign number to check:"
read CAMPAIGN_NUM
echo "Are you sure you mean campaign ${CAMPAIGN_NUM}? (y/n)"
read DECISION
if [ $DECISION == "n" ]; then
	exit 1
fi

# Path to csv file of Data ID's
ID_FILE_PATH="../../scripts/k2sc_c${CAMPAIGN_NUM}_id.csv"

# Size of file to check against (400kB)
TESTSIZE=400000

while IFS=, read -r id
do
	# Name of File to download. e.g. hlsp_k2sc_k2_llc_200008644-c05_kepler_v2_lc.fits
	FILE="hlsp_k2sc_k2_llc_${id:4:9}-c0${CAMPAIGN_NUM}_kepler_v2_lc.fits"

	# If the file doesnt exist, then download it.
	if [ ! -f $FILE ]; then
		curl -O https://archive.stsci.edu/hlsps/k2sc/v2/c0${CAMPAIGN_NUM}/${id:4:4}00000/${FILE}
	fi

	FILESIZE=$(stat -f%z "$FILE")
	# Now check that file is of sufficient size
	if [ $FILESIZE -lt $TESTSIZE ]; then
		echo "File less than testsize."
		echo "Re-downloading file: ${FILE}"
		curl -O https://archive.stsci.edu/hlsps/k2sc/v2/c0${CAMPAIGN_NUM}/${id:4:4}00000/${FILE}
	fi
done <$ID_FILE_PATH
echo "Download of campaign ${CAMPAIGN_NUM} checked and complete."

