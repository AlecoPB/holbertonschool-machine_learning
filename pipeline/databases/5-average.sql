-- compute the average of all records
ALTER TABLE second_table
ADD COLUMN average FLOAT;
UPDATE second_table
SET average = AVG(score);
