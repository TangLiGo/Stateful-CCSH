init:
	pip install -r requirements.txt

test:
	pytest

generate_md:
	jupyter nbconvert ./notebooks/hash_test.ipynb  --to markdown --output hash_test.md 

formatter:
	black thash
