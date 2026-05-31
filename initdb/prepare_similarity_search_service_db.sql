-- prepare_similarity_search_service_db.sql

-- Create the database for the similarity search service
CREATE DATABASE similarity_search_service_db;

-- Connect to the database
-- TODO: switch the psql session into similarity_search_service_db
--       (hint: use the `\c <db_name>` meta-command — same family as `\dx` from step 4)
\c similarity_search_service_db;

-- Enable vectorscale extension
-- TODO: enable the `vectorscale` extension inside the new database
--       (hint: same `CREATE EXTENSION IF NOT EXISTS <name> CASCADE;` you ran in step 5)
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;
