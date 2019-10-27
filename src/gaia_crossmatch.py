from kepler_data_utilities import *

# Example multiple entry EPICS from campaign 3.
test_epics = [205898099, 205905261, 205906121, 205908778, 205910844, 205912245,
              205926404, 205940923, 205941422]

gaia_dir = '/Users/chancehaycock/dev/machine_learning/px402/cross_match_data/gaia' 

# Some table entries appear to have multiple entries for the same EPIC. Some
# EPICs appear in more than one campaign. Also, some rows are much more heavily
# populated than others. The exact reasoning is not completely known at this
# moment in time. The below function creates a csv file unique entries chosen on
# availability. Search radius is either 1 or 20 arcsec. Call
# make_unique_gaia_csv(1) for 1 arcsec radius. Returns number of unique EPICS.
def make_unique_gaia_csv(search_radius, campaign_num):
	# Open file from 1 arsec search radius
	with fits.open('{}/k2_dr2_{}arcsec.fits'\
	               .format(gaia_dir, search_radius)) as data:
		print("Fits file imported.")
		df = pd.DataFrame(data[1].data)
		with open('{}/unique_gaia_campaign_{}_data.csv'\
		          .format(gaia_dir, campaign_num), 'a+') as file:
			# Initialise new dataframe object with same columns.
			print("Columns added to CSV File.")
			pd.DataFrame(columns=df.columns).to_csv(file, index=None)
			df = df[df['k2_campaign_str'] == '{}'.format(campaign_num)]
			# Makes them unique
			epics = set(df['epic_number'])
			epic_count = 0
			for epic in epics:
				# Do unique stuff here.
				epic_df = df[df['epic_number'] == epic]
				epic_df_size = len(epic_df.index)
				if (epic_df_size == 1):
					epic_df.to_csv(file, header=False, index=None)
					print("Added EPIC {} to CSV File. Was unique.".format(epic))
					epic_count += 1
					continue
				else:
					# Now chooses unique row fro a particular epic by looking at
					# the GAIA mean magnitude.
					temp_df = pd.DataFrame(columns=df.columns)
					gaia_mags_df = epic_df['phot_g_mean_mag']
					wanted_index = gaia_mags_df.idxmin() 
					temp_df = temp_df.append(epic_df.loc[wanted_index])
					temp_df.to_csv(file, header=False, index=None)
					print("Added EPIC {} to CSV File. Was not unique.".format(epic))
					epic_count += 1
			print("Complete.")
	return epic_count

def gaia_data_to_csv(search_radius, campaign_num):
	# Open file from 1 arsec search radius
	with fits.open('{}/gaia/k2_dr2_{}arcsec.fits'\
	               .format(gaia_dir, search_radius)) as data:
		print("Fits file imported.")
		df = pd.DataFrame(data[1].data)
		with open('{}/gaia/original_gaia_campaign_{}_data.csv'\
		          .format(gaia_dir, campaign_num), 'a+') as file:
			pd.DataFrame(columns=df.columns).to_csv(file, index=None)
			print("Reducing dataframe...")
			df = df[df['k2_campaign_str'] == '{}'.format(campaign_num)]
			df.to_csv(file, header=False, index=None)
			print("Dataframe for campaign {} converted to CSV.".format(campaign_num))
	return

def main():
	# Make 3 CSV's for campaigns 3-5
#	count3 = make_unique_gaia_csv(1, 3)
#	count4 = make_unique_gaia_csv(1, 4)
#	count5 = make_unique_gaia_csv(1, 5)
#	print("Unique EPICS from Campaign 3: ", count3)
#	print("Unique EPICS from Campaign 4: ", count4)
#	print("Unique EPICS from Campaign 5: ", count5)

	# Use this function to check original GAIA data from a specific campaign.
	# i.e. not unique.
#	gaia_data_to_csv(1, 3)
	print("Compiled.")

if __name__ == '__main__':
	main()
