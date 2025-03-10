# for scrapping data from fbref
import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import random
import os

def scrape_fbref_table(url):
    """
    Scrape a specific table from FBref URL
    
    Args:
        url (str): The URL of the FBref page to scrape
        
    Returns:
        pandas.DataFrame: DataFrame containing the scraped table
    """
    # Set headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Add delay to avoid hitting rate limits
    time.sleep(random.uniform(3, 5))
    
    # Make the request
    print(f"Scraping data from: {url}")
    response = requests.get(url, headers=headers)
    
    # Check if request was successful
    if response.status_code != 200:
        print(f"Failed to retrieve the page: {response.status_code}")
        return None
    
    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the main stats table (usually the first large table)
    table = soup.find('table', {'id': lambda x: x and x.startswith('stats_')})
    
    if not table:
        print("No table found on the page")
        return None
    
    # Convert table to pandas DataFrame
    try:
        df = pd.read_html(str(table))[0]
        
        # Clean column names
        if isinstance(df.columns, pd.MultiIndex):
            # For multi-level column headers, join them with underscores
            df.columns = ['_'.join(col).strip() if col[1] else col[0].strip() 
                        for col in df.columns.values]
        
        # Remove % symbol from column names (if present)
        df.columns = [col.replace('%', 'Pct').replace('\\', '').replace('/', '_') 
                      for col in df.columns]
        
        # Drop duplicate columns that might occur from multi-level headers
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Drop rows that are actually header rows (Rk = Rk)
        if 'Rk' in df.columns:
            df = df[df.Rk != 'Rk']
        
        # Reset index after filtering
        df = df.reset_index(drop=True)
        
        return df
    
    except Exception as e:
        print(f"Error parsing table: {e}")
        return None

def get_standard_stats():
    """
    Get standard statistics for players in top 5 European leagues
    
    Returns:
        pandas.DataFrame: DataFrame containing standard player stats
    """
    url = "https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats"
    return scrape_fbref_table(url)

def get_passing_stats():
    """
    Get passing statistics for players in top 5 European leagues
    
    Returns:
        pandas.DataFrame: DataFrame containing passing player stats
    """
    url = "https://fbref.com/en/comps/Big5/passing/players/Big-5-European-Leagues-Stats"
    return scrape_fbref_table(url)

def get_defensive_stats():
    """
    Get defensive statistics for players in top 5 European leagues
    
    Returns:
        pandas.DataFrame: DataFrame containing defensive player stats
    """
    url = "https://fbref.com/en/comps/Big5/defense/players/Big-5-European-Leagues-Stats"
    return scrape_fbref_table(url)

def get_gca_stats():
    """
    Get goal and shot creation statistics for players in top 5 European leagues
    
    Returns:
        pandas.DataFrame: DataFrame containing goal and shot creation player stats
    """
    url = "https://fbref.com/en/comps/Big5/gca/players/Big-5-European-Leagues-Stats"
    return scrape_fbref_table(url)

def save_data(data, filename):
    """
    Save data to CSV file
    
    Args:
        data: Data to save (DataFrame)
        filename (str): Name of the file to save
    """
    if data is None:
        print(f"No data to save for {filename}")
        return
        
    # Create directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    file_path = os.path.join('data', filename)
    
    data.to_csv(f"{file_path}.csv", index=False)
    print(f"Data saved to {file_path}.csv")

def scrape_all_stats():
    """
    Scrape all player statistics and save to individual CSV files
    """
    # Get all stats
    standard_stats = get_standard_stats()
    passing_stats = get_passing_stats()
    defensive_stats = get_defensive_stats()
    gca_stats = get_gca_stats()
    
    # Save individual stat files
    save_data(standard_stats, "standard_stats")
    save_data(passing_stats, "passing_stats")
    save_data(defensive_stats, "defensive_stats")
    save_data(gca_stats, "gca_stats")

# Example usage
if __name__ == "__main__":
    # Scrape all stats
    scrape_all_stats()
    
    # Or scrape individual stat categories if needed
    # standard_stats = get_standard_stats()
    # save_data(standard_stats, "standard_stats")
    
    # passing_stats = get_passing_stats()
    # save_data(passing_stats, "passing_stats")
    
    # defensive_stats = get_defensive_stats()
    # save_data(defensive_stats, "defensive_stats")
    
    # gca_stats = get_gca_stats()
    # save_data(gca_stats, "gca_stats")



