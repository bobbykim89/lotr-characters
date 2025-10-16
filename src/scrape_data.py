import pandas as pd
import json
import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
import unicodedata

character_csv = pd.read_csv('../assets/characters_with_link.csv', encoding='utf-8')
character_csv.to_json('../assets/characters_with_link.json', orient='records', indent=4)

char_detail_csv = pd.read_csv('../assets/characters_detail.csv', encoding='utf-8')
char_detail_csv.to_json('../assets/characters_detail.json', orient='records', indent=4)

with open('../assets/characters_with_link.json', 'r') as file:
    character_json = json.load(file)

with open('../assets/characters_detail.json', 'r') as file:
    character_detail_json = json.load(file)

def format_biography(soup: BeautifulSoup)-> str | None:
    try:
        biography_section = soup.find('h2', id='Biography')
        if not biography_section:
            span_with_id = soup.find('span', id='Biography')
            if span_with_id and span_with_id.parent.name == 'h2':
                biography_section = span_with_id.parent
        if not biography_section:
            return None
        biography_content = []
        biography_current_elem = biography_section.next_sibling
        while biography_current_elem:
            # Skip text nodes that are just whitespace
            if biography_current_elem.name:
                # Stop if hit another h2
                if biography_current_elem.name == 'h2':
                    break
                text_content = biography_current_elem.get_text().strip()
                biography_content.append(text_content)
            biography_current_elem = biography_current_elem.next_sibling
        biography_text = '\n'.join(biography_content)
        return biography_text
    except Exception as e:
        print(f"Error parsing content: {e}")
        return None
    
def format_history(soup: BeautifulSoup)-> str | None:
    try:
        history_section = soup.find('h2', id='History')
        if not history_section:
            span_with_id = soup.find('span', id='History')
            if span_with_id and span_with_id.parent.name == 'h2':
                history_section = span_with_id.parent
        if not history_section:
            return None
        history_content = []
        history_current_elem = history_section.next_sibling
        while history_current_elem:
            # skip text nodes that are just whitespace
            if history_current_elem.name:
                # Stop if hit another h2
                if history_current_elem.name == 'h2':
                    break
                text_content = history_current_elem.get_text().strip()
                history_content.append(text_content)
            history_current_elem = history_current_elem.next_sibling
        history_text = '\n'.join(history_content)
        return history_text
    except Exception as e:
        print(f"Error parsing content: {e}")
        return None
    
def scrape_content(url: str)-> dict[str, str | None]:
    try:
        res = requests.get(url)
        res.raise_for_status()
        soup = BeautifulSoup(res.content, 'html.parser')

        biography_text = format_biography(soup=soup)
        history_text = format_history(soup=soup)
        
        return {
            "biography": biography_text,
            "history": history_text
        }
    except requests.RequestException as e:
        print(f"Error fetching page: {e}")
        return {
            "biography": None,
            "history": None
        }
    
    except Exception as e:
        print(f"Error parsing content: {e}")
        return {
            "biography": None,
            "history": None
        }

def normalize_name(name: str):
    # remove accents and convert to lower case for comparison
    # Normalize to NFD (decomposed form)
    normalized = unicodedata.normalize('NFD', name)
    # filter out combining characters (accents, diacritics)
    ascii_text = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
    return ascii_text.lower().strip()

def get_character_detail(name: str)-> dict | None:
    normalized_search_name = normalize_name(name)
    found_object = None

    for item in character_detail_json:
        if normalize_name(item['name']) == normalized_search_name:
            found_object = item
            break # Exit loop after finding the first match

    if found_object:
        return found_object
    else:
        print(f"No object with found with name: {name}")
        return None


def scrape_and_save_in_json(file_name: str):
    characters = []
    for character in tqdm(character_json):
        character_detail = {
            "biography": None,
            "history": None
        }
        detail = get_character_detail(name=character['Name'])
        if detail is not None:
            if character['Url'] is not None:
                character_detail = scrape_content(url=character['Url'])
            character_obj = {**detail, **character_detail}
            characters.append(character_obj)
    with open(f"../dist/{file_name}.json", "w") as file:
        json.dump(characters, file, indent=4)
    print(f"Successfully saved {len(characters)} entries to {file_name}.json")

scrape_and_save_in_json('lotr_characters')