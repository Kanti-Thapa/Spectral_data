#!/usr/bin/env python
# coding: utf-8

# In[39]:


import os
import pandas as pd

# Specify the folder path
folder_path = '/Users/kantilata/Desktop/Recent_work/2024/PN24-2/PN24-2/Selected_data/G_all'

# Change the current working directory to the specified folder_path
os.chdir(folder_path)
print("Current working directory:", os.getcwd())

# List all files in the folder
files = os.listdir(folder_path)
print(files)
print(len(files))

# Filter CSV files
csv_files = [os.path.join(folder_path, file) for file in files if file.endswith('.csv')]
print("List of CSV files:", csv_files)
print(len(csv_files))


# In[40]:


# Create an empty DataFrame to store the extracted data
result_df = pd.DataFrame()

for a in csv_files:
    try:
        # Read CSV skipping rows until actual data starts
        df = pd.read_csv(a, header=None, skiprows=4)  # Assuming comma-separated values and data starts after 4 rows of header information

        # Check if there are enough rows in the DataFrame
        if len(df) >= 3500:
            # Extract the third column for rows 8 to 3500
            third_column = df.iloc[7:3501, 2].tolist()  # Assuming the third column is at index 2

            # Add a new row in the result_df with the data from the current file
            result_df = result_df.append(pd.Series(third_column, name=os.path.basename(a).split('.')[0].strip()))
        else:
            print(f"Not enough rows in {a} to extract data between rows 8 and 3500. Total rows: {len(df)} - Skipping.")
    except pd.errors.EmptyDataError:
        print(f"Skipping {a} as it is empty.")
    except Exception as e:
        print(f"Error processing {a}: {e}")

# Check if result_df has any rows before saving to CSV
if not result_df.empty:
    # Transpose the result_df to have 13 rows
    result_df = result_df.transpose()

    # Set the maximum number of columns to display to None
    pd.set_option('display.max_columns', None)

    # Save the result_df to a new CSV file
    result_df.to_csv('output_combined.csv', index=False, header=False)
    print("Data saved to 'output_combined.csv'")
else:
    print("No data extracted.")


# In[77]:


import os
import pandas as pd

# ... (your existing code)

# Create an empty DataFrame to store the extracted data
result_df = pd.DataFrame()

# Create a separate DataFrame for CSV file names
csv_names_df = pd.DataFrame({'CSV File Names': csv_files})

for a in csv_files:
    try:
        # Read CSV skipping rows until actual data starts
        df = pd.read_csv(a, header=None, skiprows=4)  # Assuming comma-separated values and data starts after 4 rows of header information

        # Check if there are enough rows in the DataFrame
        if len(df) >= 3500:
            # Extract the third column for rows 8 to 3500
            third_column = df.iloc[7:3501, 2].tolist()  # Assuming the third column is at index 2

            # Add a new row in the result_df with the data from the current file
            result_df = result_df.append(pd.Series(third_column, name=os.path.basename(a).split('.')[0].strip()))
        else:
            print(f"Not enough rows in {a} to extract data between rows 8 and 3500. Total rows: {len(df)} - Skipping.")
    except pd.errors.EmptyDataError:
        print(f"Skipping {a} as it is empty.")
    except Exception as e:
        print(f"Error processing {a}: {e}")

# Check if result_df has any rows before saving to CSV
if not result_df.empty:
    # Transpose the result_df to have 13 rows
    result_df = result_df.transpose()

    # Concatenate the CSV file names DataFrame with the transposed DataFrame
    result_df_with_names = pd.concat([csv_names_df, result_df], axis=1)

#     # Print the row names along with the corresponding CSV file
#     for row_name in result_df_with_names.index[:-1]:
#         csv_file = os.path.join(folder_path, f"{row_name}.csv")
#         print(f"Data in row '{row_name}' corresponds to CSV file: {csv_file}")

    # Set the maximum number of columns to display to None
    pd.set_option('display.max_columns', None)

    # Save the result_df_with_names to a new CSV file
    result_df_with_names.to_excel('output_combined_with_filenames.xlsx', index=False)
    print("Data saved to 'output_combined_with_filenames.xlsx'")
else:
    print("No data extracted.")
print("Current working directory:", os.getcwd())


# In[3]:


import pandas as pd


# In[4]:


last_excel_file= pd.read_excel('/Users/kantilata/Desktop/Recent_work/2024/PN24-2/PN24-2/Selected_data/G_all/output_combined_with_filenames.xlsx')
#print(last_excel_file)

transposed_data=last_excel_file.transpose()

pd.set_option('display.max_columns', None)
#exc;ude the first row before saving
transposed_data_to_save=transposed_data.iloc[1:]
# Reset the index to give the first column a proper header
transposed_data_to_save = transposed_data_to_save.reset_index()


# Set the maximum number of columns to display to None
pd.set_option('display.max_columns', None)

transposed_data_to_save.to_excel('/Users/kantilata/Desktop/Recent_work/2024/PN24-2/PN24-2/Selected_data/G_all/final_G.xlsx',index=False)
print(transposed_data_to_save)


# In[13]:


new_data=pd.read_excel('/Users/kantilata/Desktop/Recent_work/2024/PN24-2/PN24-2/Selected_data/Final_spectra_kanti.xlsx',header=None)
print(new_data)
new_data.shape


# In[14]:


x=new_data.iloc[0,2:].values
print(x)
x.shape
x=x.astype(float)
#wavelength = pd.to_numeric(new_data.iloc[:, 0], errors='coerce')
#print(wavelength)
x.shape
print(x)


# In[15]:


# In[62]:

import numpy as np

reflectance = new_data.iloc[1:,2:].values
reflectance=reflectance.T
print(reflectance)
#reflectance_shape=np.array(reflectance).shape
reflectance.shape
#kanti=int(reflectance)


# In[17]:


import matplotlib.pyplot as plt

plt.plot(x, reflectance)


# In[19]:


fig = plt.figure()
plt.figure(constrained_layout=True)
plt.rcParams["figure.figsize"] = [15,15]
import math
import matplotlib as mpl

ax1=plt.subplot()
ax1 = plt.subplot()
ax1.tick_params(axis='both',which='major', labelsize=30)
#ax1.plot(range(1), range(1), linewidth=8,color='black')
ax1.tick_params(length=10, width=2,color='black')

mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['axes.linewidth'] = 3
#plt.rc('font', weight='bold')
#plt.rcParams.update({'font.size': 100,})

plt.plot(x, reflectance,linewidth=6)
ax1.set_xlim([500,950])
ax1.set_ylim([0.01,0.8])
#plt.imshow(reflectance, cmap=plt.get_cmap('spectral'), aspect='auto')

#plt.axis('equal')
#plt.show()
#plt.title('Random Forest',size=90,color='black',fontweight='bold')
plt.xlabel('Wavelength(nm)',c='black',fontsize=35,)
plt.ylabel('Reflectance',color='black',fontsize=35)


plt.savefig('N_trifecta_F.svg')


# In[64]:




# In[ ]:





# In[ ]:





# In[ ]:




