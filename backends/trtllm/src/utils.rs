///
/// Extract the first line of the provided string reference.
/// If there is no lines in the buffer, it returns a string
/// which content is defined by the content of `fail`
/// # Arguments
///
/// * `s`: The string buffer to extract the first-line from
/// * `fail`: A string content which is returned if no lines are
/// present in `s`
///
/// returns: String
///
/// # Examples
///
/// ```
/// let s = "My name is Morgan.\n I'm working at Hugging Face.";
/// first_line(s, "No line in string");
/// ```
#[inline]
pub(crate) fn first_line(s: &str, fail: &str) -> String {
    s.lines().next().unwrap_or(fail).to_string()
}
