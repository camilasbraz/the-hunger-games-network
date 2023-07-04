# the-hunger-games-network

This is a project that uses web scrapping and natural language processing to analyze the relationship between the characters in The Hunger Games book series.

This project was inspired by the <a href = "https://github.com/thu-vu92/the_witcher_network">Thu-vu92 The witcher network</a> repo.

This analysis has two main goals:

- Investigate the characters in The Hunger Games book series regarding:

  - Importance
  - Evolution

- Investigate the communities in The Hunger Games book series.

There are three main notebooks:

- `character_extraction.ipynb` that scrappes the web to collect the characters name and save it into a csv file

1. Sets up Chrome options to run in headless mode and with no sandbox.

2. Creates a Chrome WebDriver service and a WebDriver instance using the configured options.

3. Navigates to the "Characters" page on the Hunger Games Fandom website.

4. Accepts cookies by clicking on the "ACEITAR" button.

5. Scrapes information about books and characters from specific categories on the page, stores the data in a DataFrame, and saves it as a CSV file named "characters.csv". Additionally, it generates a bar plot of character counts per book.

- `character_refinement.ipynb` that was used to improve the quality of the data regarding characters names. 
The analysis made without this refinement does not produces good results (garbage-in-garbage-out)
1. 

- `network-final.ipynb` that 
1. 

- `evolution.ipynb` that creates graphs representing character relationships based on co-occurrence in the text, calculates the degree centrality of each character in each book, and plots the degree centrality evolution of five main characters across the books.

1. Load the necessary modules and libraries, including spacy, pandas, networkx, and regular expressions.

2. Retrieve a list of PDF books from a specified directory and sort them by name.

3. Iterate through each book and perform the following actions:

      a. Extract character entities using NER and filter out non-character entities.

      b. Clean and process the character names.

      c. Create a dataframe with relationships between characters based on their co-occurrence in sentences.

      d. Build a graph from the dataframe using networkx.

4. Store each graph in a list for further analysis.

5. Calculate the degree centrality for each character in each book's graph and plot the evolution of degree centrality for five main characters across all books.

- `functions.py` contains functions that are used in the code  to process the text from PDF files, extract named entities per sentence, filter character entities and create relationship dataframes.

1. `ner(file_path, start_page)`: This function takes a PDF file path and a starting page number as input. It uses the pdfplumber library to extract the text content from the specified page onwards. Then, it uses the spaCy library to perform named entity recognition (NER) on the extracted text. The processed Doc object is returned.

2. `get_ne_list_per_sentence(spacy_doc)`: This function takes a processed spaCy Doc object as input. It iterates through the sentences in the document and stores the named entities for each sentence in a dataframe. The resulting dataframe contains two columns: "sentence" (the sentence itself) and "entities" (the list of named entities in the sentence).

3. `filter_entity(ent_list, character_df)`: This function filters out non-character entities from a list of entities. It takes the list of entities to be filtered and a dataframe (character_df) containing characters' names and first names. It checks if each entity is present in either the "character" column or the "character_firstname" column of the dataframe. Only the entities that match the character names or first names are returned.

4. `create_relationships(df, window_size)`: This function creates a dataframe of relationships between characters based on the input dataframe (df) and a window size (window_size). It iterates through the rows of the input dataframe, considering a window of adjacent sentences defined by the window size. For each window, it collects the unique characters and creates relationships between adjacent characters. The resulting relationship dataframe contains three columns: "source" (the source character), "target" (the target character), and "value" (the number of relationships between the characters).


The results are available <a href = "">here</a>
