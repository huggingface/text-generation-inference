Documentation available at: https://huggingface.co/docs/text-generation-inference

## Release

When making a release, please update the latest version in the documentation with:
```
export OLD_VERSION="2\.0\.3"
export NEW_VERSION="2\.0\.4"
find . -name '*.md' -exec sed -i -e "s/$OLD_VERSION/$NEW_VERSION/g" {} \;
```
