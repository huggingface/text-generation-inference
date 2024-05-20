
build:
	maturin build --sdist --interpreter python3.11

# by commenting out the line in .gitignore, we can include the generated files in the git repository
comment-gitignore:
	sed -i 's/^\(.*\)/# \1/' ../server/text_generation_server/pb/.gitignore

remove-comment-gitignore:
	sed -i 's/^# \(.*\)/\1/' ../server/text_generation_server/pb/.gitignore

comment-gitignore2:
	sed -i 's/^\(.*\)/# \1/' ../router/client/src/pb/.gitignore

remove-comment-gitignore2:
	sed -i 's/^# \(.*\)/\1/' ../router/client/src/pb/.gitignore


library-install:
	pip install -e .

install: comment-gitignore comment-gitignore2 build library-install remove-comment-gitignore remove-comment-gitignore2

quick-install: build library-install
