<!---
Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Contribute to text-generation-inference

Everyone is welcome to contribute, and we value everybody's contribution. Code
contributions are not the only way to help the community. Answering questions, helping
others, and improving the documentation are also immensely valuable.

It also helps us if you spread the word! Reference the library in blog posts
about the awesome projects it made possible, shout out on Twitter every time it has
helped you, or simply ⭐️ the repository to say thank you.

However you choose to contribute, please be mindful and respect our
[code of conduct](https://github.com/huggingface/text-generation-inference/blob/main/CODE_OF_CONDUCT.md).

**This guide was heavily inspired by the awesome [scikit-learn guide to contributing](https://github.com/scikit-learn/scikit-learn/blob/main/CONTRIBUTING.md).**

## Ways to contribute

There are several ways you can contribute to text-generation-inference.

* Fix outstanding issues with the existing code.
* Submit issues related to bugs or desired new features.
* Contribute to the examples or to the documentation.

> All contributions are equally valuable to the community. 🥰

## Fixing outstanding issues

If you notice an issue with the existing code and have a fix in mind, feel free to [start contributing](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) and open
a Pull Request!

## Submitting a bug-related issue or feature request

Do your best to follow these guidelines when submitting a bug-related issue or a feature
request. It will make it easier for us to come back to you quickly and with good
feedback.

### Did you find a bug?

The text-generation-inference library is robust and reliable thanks to users who report the problems they encounter.

Before you report an issue, we would really appreciate it if you could **make sure the bug was not
already reported** (use the search bar on GitHub under Issues). Your issue should also be related to bugs in the
library itself, and not your code.

Once you've confirmed the bug hasn't already been reported, please include the following information in your issue so
we can quickly resolve it:

* Your **OS type and version**, as well as your environment versions (versions of rust, python, and dependencies).
* A short, self-contained, code snippet that allows us to reproduce the bug.
* The *full* traceback if an exception is raised.
* Attach any other additional information, like screenshots, you think may help.

To get the OS and software versions automatically, you can re-run the launcher with the `--env` flag:

```bash
text-generation-launcher --env
```

This will precede the launch of the model with the information relative to your environment. We recommend pasting
that in your issue report.

### Do you want a new feature?

If there is a new feature you'd like to see in text-generation-inference, please open an issue and describe:

1. What is the *motivation* behind this feature? Is it related to a problem or frustration with the library? Is it
   a feature related to something you need for a project? Is it something you worked on and think it could benefit
   the community?

   Whatever it is, we'd love to hear about it!

2. Describe your requested feature in as much detail as possible. The more you can tell us about it, the better
   we'll be able to help you.
3. Provide a *code snippet* that demonstrates the feature's usage.
4. If the feature is related to a paper, please include a link.

If your issue is well written we're already 80% of the way there by the time you create it.

We have added [templates](https://github.com/huggingface/text-generation-inference/tree/main/.github/ISSUE_TEMPLATE)
to help you get started with your issue.

## Do you want to implement a new model?

New models are constantly released and if you want to implement a new model, please provide the following information:

* A short description of the model and a link to the paper.
* Link to the implementation if it is open-sourced.
* Link to the model weights if they are available.

If you are willing to contribute the model yourself, let us know so we can help you add it to text-generation-inference!

## Do you want to add documentation?

We're always looking for improvements to the documentation that make it more clear and accurate. Please let us know
how the documentation can be improved such as typos and any content that is missing, unclear or inaccurate. We'll be
happy to make the changes or help you make a contribution if you're interested!

## I want to become a maintainer of the project. How do I get there?

TGI is a project led and managed by Hugging Face as it powers our internal services. However, we are happy to have
motivated individuals from other organizations join us as maintainers with the goal of making TGI the best inference
service.

If you are such an individual (or organization), please reach out to us and let's collaborate.
