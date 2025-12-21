import pandas as pd

# Read the venue mapping
venues_df = pd.read_csv('../data/old/mlb_venues.csv')

# Read the park factors
park_factors_df = pd.read_csv('../data/old/mlb_park_factors.csv')

# Merge on venue name
merged_df = venues_df.merge(
    park_factors_df,
    on='venue_name',
    how='left'
)

# Fill missing park factors with neutral value (100)
merged_df['park_factor'] = merged_df['park_factor'].fillna(100)

# Sort by venue_id
merged_df = merged_df.sort_values('venue_id')

# Save the merged data
merged_df.to_csv('../data/venue_park_factors.csv', index=False)

print("Merged venue park factors:")
print(merged_df)

# Check for any venues without park factors
missing = merged_df[merged_df['park_factor'] == 100]
if len(missing) > 0:
    print("\nVenues with neutral park factor (100):")
    print(missing[['venue_id', 'venue_name']])
