use std::collections::BTreeMap;

// TODO
#[allow(dead_code)]
#[cfg_attr(test, derive(Debug, PartialEq))]
pub enum Error {
    MissingEntry,
}

#[derive(Clone)]
pub struct Trie {
    root: Node,
}

#[derive(Clone)]
#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Node {
    content: Vec<u8>,
    nelements: usize,
    local_elements: usize,
    children: BTreeMap<u8, Node>,
}

pub fn mismatch(xs: &[u8], ys: &[u8]) -> usize {
    // SIMD
    mismatch_chunks::<128>(xs, ys)
}

fn mismatch_chunks<const N: usize>(xs: &[u8], ys: &[u8]) -> usize {
    let off = xs
        .chunks_exact(N)
        .zip(ys.chunks_exact(N))
        .take_while(|(x, y)| x == y)
        .count()
        * N;
    off + xs[off..]
        .iter()
        .zip(&ys[off..])
        .take_while(|(x, y)| x == y)
        .count()
}

impl Node {
    fn new() -> Self {
        Self {
            content: vec![],
            nelements: 0,
            local_elements: 0,
            children: BTreeMap::new(),
        }
    }

    fn insert(&mut self, data: &[u8]) -> (usize, usize) {
        let (start, stop) = if self.nelements == 0 {
            self.content = data.to_vec();
            assert_eq!(self.local_elements, 0);
            self.local_elements = 1;
            (0, self.local_elements)
        } else {
            let mismatch = mismatch(data, &self.content);
            if mismatch == self.content.len() {
                // Full prefix match, just dive deeper
                let (start, stop) = if let Some(c) = data.get(mismatch) {
                    let left: usize = self
                        .children
                        .iter()
                        .take_while(|(&d, _)| d < *c)
                        .map(|(_, n)| n.nelements)
                        .sum::<usize>()
                        + self.local_elements;
                    let next_node = self.children.entry(*c).or_insert(Node::new());
                    let (inner_left, inner_right) = next_node.insert(&data[mismatch..]);
                    (inner_left + left, inner_right + left)
                } else {
                    assert_eq!(data.len(), self.content.len());
                    self.local_elements += 1;
                    (0, self.local_elements)
                };
                (start, stop)
            } else {
                // Partial match, split node
                let left_content = self.content[mismatch..].to_vec();
                let right_content = data[mismatch..].to_vec();
                let children = std::mem::take(&mut self.children);
                let mut children_content = vec![
                    (left_content, children, self.nelements, self.local_elements),
                    (right_content, BTreeMap::new(), 1, 1),
                ];
                children_content.sort_by(|a, b| a.0.cmp(&b.0));
                self.content.truncate(mismatch);
                self.children.clear();
                self.local_elements = 0;
                for (child_content, children, nelements, local_elements) in children_content {
                    if !child_content.is_empty() {
                        let c = child_content[0];
                        let child = Node {
                            content: child_content,
                            nelements,
                            local_elements,
                            children,
                        };
                        self.children.insert(c, child);
                    } else {
                        self.local_elements += 1;
                    }
                }
                let (start, stop) = if let Some(c) = data.get(mismatch) {
                    let start = self
                        .children
                        .iter()
                        .take_while(|(&d, _)| d < *c)
                        .map(|(_, n)| n.nelements)
                        .sum::<usize>()
                        + self.local_elements;
                    (start, start + 1)
                } else {
                    (0, self.local_elements)
                };
                (start, stop)
            }
        };
        self.nelements += 1;
        (start, stop)
    }

    #[cfg(debug_assertions)]
    fn remove(&mut self, data: &[u8]) -> Result<(), Error> {
        // TODO reclaim the nodes too.
        let mismatch = mismatch(data, &self.content);
        if mismatch != self.content.len() {
            Err(Error::MissingEntry)
        } else {
            if let Some(c) = data.get(mismatch) {
                if let Some(node) = self.children.get_mut(c) {
                    node.remove(&data[mismatch..])?;
                }
            } else if self.local_elements == 0 {
                return Err(Error::MissingEntry);
            } else {
                self.local_elements -= 1;
            }
            self.nelements -= 1;
            Ok(())
        }
    }
}

impl Trie {
    pub fn new() -> Self {
        let root = Node::new();
        Self { root }
    }

    pub fn insert(&mut self, data: &[u8]) -> (usize, usize) {
        self.root.insert(data)
    }

    // TODO
    #[allow(dead_code)]
    pub fn remove(&mut self, data: &[u8]) -> Result<(), Error> {
        self.root.remove(data)
    }

    pub fn count(&self) -> usize {
        self.root.nelements
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple() {
        let mut trie = Trie::new();
        assert_eq!(trie.insert(b"toto"), (0, 1));
        assert_eq!(trie.insert(b"tata"), (0, 1));

        assert_eq!(trie.root.nelements, 2);
        assert_eq!(trie.root.content, b"t");
        assert_eq!(trie.root.local_elements, 0);
        assert_eq!(trie.root.children.len(), 2);
        assert_eq!(
            trie.root.children,
            BTreeMap::from_iter([
                (
                    b'a',
                    Node {
                        nelements: 1,
                        local_elements: 1,
                        content: b"ata".to_vec(),
                        children: BTreeMap::new()
                    }
                ),
                (
                    b'o',
                    Node {
                        nelements: 1,
                        local_elements: 1,
                        content: b"oto".to_vec(),
                        children: BTreeMap::new()
                    }
                )
            ])
        );
        assert_eq!(trie.insert(b"coco"), (0, 1));
        assert_eq!(trie.insert(b"zaza"), (3, 4));
        assert_eq!(trie.root.nelements, 4);
        assert_eq!(trie.root.local_elements, 0);
        assert_eq!(trie.root.content, b"");
        assert_eq!(trie.root.children.len(), 3);
        assert_eq!(
            trie.root.children,
            BTreeMap::from_iter([
                (
                    b'c',
                    Node {
                        nelements: 1,
                        local_elements: 1,
                        content: b"coco".to_vec(),
                        children: BTreeMap::new()
                    }
                ),
                (
                    b't',
                    Node {
                        nelements: 2,
                        local_elements: 0,
                        content: b"t".to_vec(),
                        children: BTreeMap::from_iter([
                            (
                                b'a',
                                Node {
                                    nelements: 1,
                                    local_elements: 1,
                                    content: b"ata".to_vec(),
                                    children: BTreeMap::new()
                                }
                            ),
                            (
                                b'o',
                                Node {
                                    nelements: 1,
                                    local_elements: 1,
                                    content: b"oto".to_vec(),
                                    children: BTreeMap::new()
                                }
                            )
                        ])
                    }
                ),
                (
                    b'z',
                    Node {
                        nelements: 1,
                        local_elements: 1,
                        content: b"zaza".to_vec(),
                        children: BTreeMap::new()
                    }
                ),
            ])
        );
    }

    #[test]
    fn delete() {
        let mut trie = Trie::new();
        trie.insert(b"toto");
        trie.insert(b"tata");

        assert_eq!(trie.root.nelements, 2);
        assert_eq!(trie.remove(b"coco"), Err(Error::MissingEntry));
        assert_eq!(trie.remove(b"toto"), Ok(()));
        assert_eq!(trie.root.nelements, 1);
    }

    #[test]
    fn delete_prefix() {
        let mut trie = Trie::new();
        trie.insert(b"toto");
        trie.insert(b"to");

        assert_eq!(trie.root.nelements, 2);
        assert_eq!(trie.remove(b"to"), Ok(()));
        assert_eq!(trie.root.nelements, 1);
        assert_eq!(trie.remove(b"toto"), Ok(()));
        assert_eq!(trie.root.nelements, 0);
    }

    #[test]
    fn duplicate() {
        let mut trie = Trie::new();
        assert_eq!(trie.insert(b"toto"), (0, 1));
        assert_eq!(trie.insert(b"toto"), (0, 2));
        assert_eq!(trie.root.nelements, 2);
        assert_eq!(trie.remove(b"toto"), Ok(()));
        assert_eq!(trie.root.nelements, 1);
    }

    #[test]
    fn prefix() {
        let mut trie = Trie::new();
        assert_eq!(trie.insert(b"toto"), (0, 1));
        assert_eq!(trie.insert(b"to"), (0, 1));
        assert_eq!(trie.root.nelements, 2);
        assert_eq!(trie.insert(b"toto"), (1, 3));
        assert_eq!(trie.root.nelements, 3);
        assert_eq!(trie.insert(b"tototo"), (3, 4));
        assert_eq!(trie.root.nelements, 4);
        assert_eq!(trie.remove(b"toto"), Ok(()));
        assert_eq!(trie.root.nelements, 3);
    }

    #[test]
    fn test_mismatch() {
        let m = mismatch(&[0, 1, 2], &[0, 1, 3]);
        assert_eq!(m, 2);
        let a = vec![0; 256];
        let mut b = vec![0; 256];
        assert_eq!(mismatch(&a, &b), 256);
        b[130] = 1;
        assert_eq!(mismatch(&a, &b), 130);
        b[129] = 1;
        assert_eq!(mismatch(&a, &b), 129);
        b[128] = 1;
        assert_eq!(mismatch(&a, &b), 128);
    }
}
