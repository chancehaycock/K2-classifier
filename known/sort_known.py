# Sorts data from Campaigns 0-4 into separate files with name_structure
# {campaign_num}_{Star Type}

# Known Types:
# - Noise
# - OTHPER
# - EA (Eclipsing Binary)
# - DSCUT
# - EB (Eclipsing Binary)
# - GDOR
# - RRab

import csv

def write_to_file(kepler_id, campaign_num, star_type):
	fout = open("{}_{}.txt".format(campaign_num, star_type), "a+")
	fout.write("{}\n".format(kepler_id))
	fout.close()

with open("armstrong_0_to_4.csv") as knownfile:
	reader = csv.DictReader(knownfile)
	for row in reader:
		star_type = row["Class"].split()[0]
		kepler_id = row["epic_number"]
		campaign_num = row["Campaign"]
		write_to_file(kepler_id, campaign_num, star_type)


