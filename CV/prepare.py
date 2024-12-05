## Import packages
import os
import sys
import pandas as pd
import hashlib

## Compute md5 hash of images to identify and drop duplicates
def hash_file(file):
    with open(file, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def main():
    if (len(sys.argv) != 2):
        print("Usage: python prepare.py data_dir")
        return 1

    ## Identify image filepaths in data directory
    data_dir = sys.argv[1]
    output_fname = data_dir + '.csv'

    category_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    categories = [os.path.basename(d) for d in category_dirs]
    print(data_dir)
    print(categories)

    food_image_files = {}

    for category in categories:
        food_image_files[category] = [os.path.join(data_dir, category, f) for f in os.listdir(os.path.join(data_dir, category)) if os.path.isfile(os.path.join(data_dir, category, f))]

    # print(food_image_files)

    ## Identify number of images in each category
    for category in categories:
        print(category, len(food_image_files[category]))

    ## Place image filepaths in dictionary into data frame with category as label
    food_image_df = pd.DataFrame(columns=['label', 'file'])

    for category in categories:
        for file in food_image_files[category]:
            food_image_df = food_image_df._append({'label': category, 'file': file}, ignore_index=True)

    food_image_df['hash'] = food_image_df['file'].apply(hash_file)

    ## Identify and remove duplicates
    # duplicates = food_image_df[food_image_df.duplicated(subset='hash', keep=False)]
    # print(duplicates)
    print(len(food_image_df))
    food_image_df = food_image_df.drop_duplicates(subset='hash')

    print(len(food_image_df))

    # Drop hash column
    food_image_df = food_image_df.drop(columns=['hash'])

    ## Save data frame to csv
    food_image_df.to_csv(output_fname, index=False)

    return 0


if __name__ == '__main__':
    exit(main())