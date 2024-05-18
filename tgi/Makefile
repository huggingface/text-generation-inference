
build:
	maturin build

# by commenting out the line in .gitignore, we can include the generated files in the git repository
comment-gitignore:
	sed -i 's/^\(.*\)/# \1/' ../server/text_generation_server/pb/.gitignore

remove-comment-gitignore:
	sed -i 's/^# \(.*\)/\1/' ../server/text_generation_server/pb/.gitignore

library-install:
	pip install -e .

install: build comment-gitignore library-install remove-comment-gitignore

quick-install: build library-install
