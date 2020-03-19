# Neural Network Attempt

from kepler_data_utilities import *

import tensorflow as tf

def NN():
	train_file = "{}/training_sets/k2sc/c1-4_gamma.csv".format(project_dir)
	train_epics = pd.read_csv(train_file)['epic_number'].to_numpy()
	print(train_epics)

	data_file_1 = "{}/phasefold_bins/k2sc/campaign_1_interpolated.csv".format(project_dir)
	data_file_2 = "{}/phasefold_bins/k2sc/campaign_2_interpolated.csv".format(project_dir)
	data_file_3 = "{}/phasefold_bins/k2sc/campaign_3_interpolated.csv".format(project_dir)
	data_file_4 = "{}/phasefold_bins/k2sc/campaign_4_interpolated.csv".format(project_dir)
	data_df_1 = pd.read_csv(data_file_1)
	data_df_2 = pd.read_csv(data_file_2)
	data_df_3 = pd.read_csv(data_file_3)
	data_df_4 = pd.read_csv(data_file_4)
	data_df = data_df_1.append(data_df_2).append(data_df_3).append(data_df_4)
	data_df = data_df[data_df['epic_number'].isin(train_epics)]
	nn_data = data_df.drop('epic_number', axis=1).to_numpy()
	print(nn_data)
	print(nn_data.shape)

	model = tf.keras.models.Sequential([
	        tf.keras.layers.Flatten(input_shape=nn_data.shape),
	        tf.keras.layers.Dense(128, activation='relu'),
	        tf.keras.layers.Dropout(0.2),
	        tf.keras.layers.Dense(7)
	        ])
	model.compile(optimizer='adam',
           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	       metrics=['accuracy'])


def main():
	NN()

if __name__ == "__main__":
	main()

