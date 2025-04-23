#!/usr/bin/env python3
# coding: utf-8


import json
import polars as pl
import logging


def main():

    metadata={'university_students_mental_health':{},
    'student_admission_record':{},
    'student_information_dataset':{},
    'customer_shopping_dataset':{},
    'student_lifestyle_dataset':{},
    'gym_members_exercise_dataset':{},
    'linkedIn_Influencers_data':{},
    'adult_census_income':{},
    'california_housing_prices':{},
    'contraceptive_method_choice':{},
    'mammographic_mass_dataset':{}}

    for i in range(1,6):
        print(f'categories/categories_0{i}.xlsx')
        df = pl.read_excel(source=f'categories/categories_0{i}.xlsx',
        has_header=True,
        sheet_name='datasets')
        data = df.rows()
        print(f'{df}')
        for l in data:
            feature = l[1].strip()
            if feature not in metadata[l[0]]:
                metadata[l[0]][feature] = {}
            categories = l[2].split(';')
            for category in categories:
                category = category.strip()
                if category:
                    if category not in metadata[l[0]][feature]:
                        metadata[l[0]][feature][category] = 0
                    metadata[l[0]][feature][category] += 1

    #Normalize data
    for dataset_name in metadata:
        dataset=metadata[dataset_name]
        for feature_name in dataset:
            feature = dataset[feature_name]
            # count total:
            t = 0.0
            for k in feature:
                t += feature[k]
            # normalize
            for k in feature:
                feature[k] /= t
    print(f'{metadata}')

    #store into metadata.json
    with open("metadata.json", "w") as outfile:
        json.dump(metadata, outfile)


if __name__ == '__main__':
    main()