"""
Country data for pre-populating the database
"""
COUNTRIES_DATA = [
    {
        "country_name": "United States",
        "flag_colors": ["Red", "White", "Blue"],
        "capital": "Washington D.C.",
        "population": "331 million",
        "continent": "North America",
        "info_type": "db"
    },
    {
        "country_name": "France",
        "flag_colors": ["Blue", "White", "Red"],
        "capital": "Paris",
        "population": "67 million",
        "continent": "Europe",
        "info_type": "db"
    },
    {
        "country_name": "Japan",
        "flag_colors": ["White", "Red"],
        "capital": "Tokyo",
        "population": "125 million",
        "continent": "Asia",
        "info_type": "db"
    },
    {
        "country_name": "Brazil",
        "flag_colors": ["Green", "Yellow", "Blue", "White"],
        "capital": "Brasília",
        "population": "215 million",
        "continent": "South America",
        "info_type": "db"
    },
    {
        "country_name": "Germany",
        "flag_colors": ["Black", "Red", "Gold"],
        "capital": "Berlin",
        "population": "83 million",
        "continent": "Europe",
        "info_type": "db"
    },
    {
        "country_name": "India",
        "flag_colors": ["Saffron", "White", "Green"],
        "capital": "New Delhi",
        "population": "1.4 billion",
        "continent": "Asia",
        "info_type": "db"
    },
    {
        "country_name": "United Kingdom",
        "flag_colors": ["Red", "White", "Blue"],
        "capital": "London",
        "population": "67 million",
        "continent": "Europe",
        "info_type": "db"
    },
    {
        "country_name": "Canada",
        "flag_colors": ["Red", "White"],
        "capital": "Ottawa",
        "population": "38 million",
        "continent": "North America",
        "info_type": "db"
    },
    {
        "country_name": "Australia",
        "flag_colors": ["Blue", "Red", "White"],
        "capital": "Canberra",
        "population": "26 million",
        "continent": "Oceania",
        "info_type": "db"
    },
    {
        "country_name": "China",
        "flag_colors": ["Red", "Yellow"],
        "capital": "Beijing",
        "population": "1.4 billion",
        "continent": "Asia",
        "info_type": "db"
    },
    {
        "country_name": "Mexico",
        "flag_colors": ["Green", "White", "Red"],
        "capital": "Mexico City",
        "population": "128 million",
        "continent": "North America",
        "info_type": "db"
    },
    {
        "country_name": "Italy",
        "flag_colors": ["Green", "White", "Red"],
        "capital": "Rome",
        "population": "59 million",
        "continent": "Europe",
        "info_type": "db"
    },
    {
        "country_name": "Spain",
        "flag_colors": ["Red", "Yellow", "Red"],
        "capital": "Madrid",
        "population": "47 million",
        "continent": "Europe",
        "info_type": "db"
    },
    {
        "country_name": "Russia",
        "flag_colors": ["White", "Blue", "Red"],
        "capital": "Moscow",
        "population": "146 million",
        "continent": "Europe/Asia",
        "info_type": "db"
    },
    {
        "country_name": "South Korea",
        "flag_colors": ["White", "Red", "Blue", "Black"],
        "capital": "Seoul",
        "population": "51 million",
        "continent": "Asia",
        "info_type": "db"
    },
    {
        "country_name": "Argentina",
        "flag_colors": ["Light Blue", "White", "Yellow"],
        "capital": "Buenos Aires",
        "population": "45 million",
        "continent": "South America",
        "info_type": "db"
    },
    {
        "country_name": "South Africa",
        "flag_colors": ["Red", "Blue", "Green", "Yellow", "White", "Black"],
        "capital": "Cape Town, Pretoria, Bloemfontein",
        "population": "60 million",
        "continent": "Africa",
        "info_type": "db"
    },
    {
        "country_name": "Egypt",
        "flag_colors": ["Red", "White", "Black", "Gold"],
        "capital": "Cairo",
        "population": "109 million",
        "continent": "Africa",
        "info_type": "db"
    },
    {
        "country_name": "Nigeria",
        "flag_colors": ["Green", "White", "Green"],
        "capital": "Abuja",
        "population": "218 million",
        "continent": "Africa",
        "info_type": "db"
    },
    {
        "country_name": "Turkey",
        "flag_colors": ["Red", "White"],
        "capital": "Ankara",
        "population": "85 million",
        "continent": "Europe/Asia",
        "info_type": "db"
    }
]

def format_country_text(country_data: dict) -> str:
    """
    Format country data into a text string for the database
    
    Args:
        country_data: Dictionary containing country information
        
    Returns:
        Formatted text string
    """
    text = f"Country: {country_data['country_name']}\n"
    text += f"Capital: {country_data['capital']}\n"
    text += f"Population: {country_data['population']}\n"
    text += f"Continent: {country_data['continent']}\n"
    text += f"Flag Colors: {', '.join(country_data['flag_colors'])}\n"
    text += f"\nThe flag of {country_data['country_name']} features the colors: {', '.join(country_data['flag_colors'])}. "
    text += f"The capital city is {country_data['capital']} and the country is located in {country_data['continent']}. "
    text += f"The population is approximately {country_data['population']}."
    
    return text








