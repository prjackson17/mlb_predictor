import statsapi
import pandas as pd


def get_mlb_venues():
    """
    Fetches all MLB venue IDs and names from statsapi.
    Saves to CSV for reference.
    """
    venue_mapping = {}

    # Get venues from teams (most reliable method)
    try:
        teams = statsapi.get('teams', {'sportId': 1})

        for team in teams.get('teams', []):
            venue = team.get('venue', {})
            venue_id = venue.get('id')
            venue_name = venue.get('name')

            if venue_id and venue_name:
                venue_mapping[venue_id] = venue_name

        print(f"Found {len(venue_mapping)} venues")

        # Convert to DataFrame
        df = pd.DataFrame(list(venue_mapping.items()),
                          columns=['venue_id', 'venue_name'])
        df = df.sort_values('venue_name')

        # Save to CSV
        output_file = "../data/old/mlb_venues.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved to {output_file}")

        # Print for reference
        print("\nVenue Mapping:")
        print(df.to_string(index=False))

        return df

    except Exception as e:
        print(f"Error fetching venues: {e}")
        return None


if __name__ == "__main__":
    get_mlb_venues()
