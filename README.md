# the-hunger-games-network

This is a project that uses web scrapping and natural language processing to analyze the relationship between the characters in The Hunger Games book series.

This project was inspired by the <a href = "https://github.com/thu-vu92/the_witcher_network">Thu-vu92 The witcher network</a> repo.

This analysis has two main goals:

- Investigate the characters in The Hunger Games book series:

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

- `network.ipynb` that 
1. 


The results are available <a href = "">here</a>
