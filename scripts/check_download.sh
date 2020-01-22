#!/bin/bash

# Choose Campaign Number to Check
echo "Enter campaign number to check:"
read CAMPAIGN_NUM
echo "Are you sure you mean campaign ${CAMPAIGN_NUM}? (y/n)"
read DECISION
if [ $DECISION == "n" ]; then
	exit 1
fi
echo "Which detrending? Everest or K2SC [e/k]?"
read DETRENDING


# Path to csv file of Data ID's
ID_FILE_PATH="../../kepler_ids/c${CAMPAIGN_NUM}_star_ids.csv"


while IFS=, read -r id
do
	# Name of File to download. e.g. hlsp_k2sc_k2_llc_200008644-c05_kepler_v2_lc.fits
	
	if [ $DETRENDING == "e" ]; then
		# Size of file to check against (10MB)
		TESTSIZE=10000000
		FILE="hlsp_everest_k2_llc_${id}-c0${CAMPAIGN_NUM}_kepler_v2.0_lc.fits"
		echo $FILE
		# If the file doesnt exist, then download it.
		if [ ! -f $FILE ]; then
#			curl -O https://archive.stsci.edu/hlsps/everest/v2/c03/205800000/89250/hlsp_everest_k2_llc_205889250-c03_kepler_v2.0_lc.fits
			curl -O https://archive.stsci.edu/hlsps/everest/v2/c0${CAMPAIGN_NUM}/${id:0:4}00000/${id:4:5}/${FILE}
		fi

	#	FILESIZE=$(stat -f%z "$FILE")
#		# Now check that file is of sufficient size
#		if [ $FILESIZE -lt $TESTSIZE ]; then
#			echo "File less than testsize."
#			echo "Re-downloading file: ${FILE}"
#			curl -O https://archive.stsci.edu/hlsps/everest/v2/c0${CAMPAIGN_NUM}/${id:4:4}00000/${FILE}
#		fi

		echo "Download of campaign ${CAMPAIGN_NUM} checked and complete."
	fi
	if [ $DETRENDING == "k" ]; then
	# Size of file to check against (400kB)
	TESTSIZE=400000
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

		echo "Download of campaign ${CAMPAIGN_NUM} checked and complete."
	fi
done <$ID_FILE_PATH

