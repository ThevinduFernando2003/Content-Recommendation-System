# Demo script

About three minutes. Run the API and the UI first (`python scripts/serve.py`).

1. Problem. We have a Netflix catalogue and no viewing history. The job is: given a title, or a sentence about taste, return similar titles and say why.
2. Data. The spreadsheet is not the tidy table it looks like. Missing director and country are the words "Not Given". A few titles were stored as Excel dates (`Feb-09`, `15-Aug`, `22-Jul`). Some text was UTF-8 read as Latin-1. `cast` and `description` are not in this file at all. The cleaner handles each of those, and it does not invent the token "Unknown".
3. Live recommendation. Search `breakng bad`. The page should say it resolved to Breaking Bad. The cards are other crime dramas, with a "Because…" line. Ozark is a fair neighbour. Dare Me is not. Say that out loud. The genre triple is the whole signal, because there is no cast and no synopsis.
4. Explanation that does work. Search The Irishman. Gangs of New York, GoodFellas, and The Departed come up because they share Martin Scorsese, and the reason line says so. Hugo appears for the same director even though it is a family film.
5. Franchise toggle. Search `Naruto Shippuden: The Movie`. With "Hide sequels and spin-offs" on, Bonds is absent. Turn it off and Bonds is first, because the titles share a long prefix and the director Hajime Kamegaki.
6. Cold start. Open "Describe your taste" and use "crime drama thriller". The scores are low. The model can only match words that exist in genres, directors, and ratings.
7. Metrics. Open "About the model". Genre precision@10 is 0.9973 against a random baseline of 0.2422. Say that this is not accuracy: genres are both the feature and the label. Coverage of top-10 lists is 0.8729. Intra-list diversity is 0.0934, which means a list of ten is often ten near-copies.
8. Architecture. CSV, clean, weighted text, TF-IDF, one sparse matrix, a dot product per request, FastAPI, Streamlit over HTTP. The UI does not import the model. The full pair matrix is not stored. Artifacts are about 2 MB.
