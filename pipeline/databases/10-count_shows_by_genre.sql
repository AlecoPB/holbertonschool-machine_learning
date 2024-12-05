-- counts how many shows belong to each genre
SELECT tv_show_genres.genre_id, COUNT(tv_show_genres.show_id) AS number_of_shows
FROM tv_show_genres
GROUP BY tv_show_genres.genre_id;
