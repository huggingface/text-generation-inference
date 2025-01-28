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
            children: BTreeMap::new(),
        }
    }

    fn insert(&mut self, data: &[u8], left: usize) -> (usize, usize) {
        let (start, stop) = if self.nelements == 0 {
            self.content = data.to_vec();
            (left, left + 1)
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
                        .sum();
                    let next_node = self.children.entry(*c).or_insert(Node::new());
                    next_node.insert(&data[mismatch..], left)
                } else {
                    (0, self.nelements + 1)
                };
                (left + start, left + stop)
            } else {
                // Partial match, split node
                let left = self.content[mismatch..].to_vec();
                let right = data[mismatch..].to_vec();

                let children = std::mem::take(&mut self.children);
                let mut children_content = vec![
                    (left, children, self.nelements),
                    (right, BTreeMap::new(), 1),
                ];
                children_content.sort_by(|a, b| a.0.cmp(&b.0));
                self.content.truncate(mismatch);
                self.children.clear();
                for (child_content, children, nelements) in children_content {
                    if !child_content.is_empty() {
                        let c = child_content[0];
                        let child = Node {
                            content: child_content,
                            nelements,
                            children,
                        };
                        self.children.insert(c, child);
                    }
                }
                let c = data[mismatch];
                let left: usize = self
                    .children
                    .iter()
                    .take_while(|(&d, _)| d < c)
                    .map(|(_, n)| n.nelements)
                    .sum();
                (left, left + 1)
            }
        };
        self.nelements += 1;
        (start, stop)
    }

    // TODO
    #[allow(dead_code)]
    fn remove(&mut self, data: &[u8]) -> Result<(), Error> {
        let mismatch = mismatch(data, &self.content);
        if mismatch != self.content.len() {
            Err(Error::MissingEntry)
        } else {
            if let Some(c) = data.get(mismatch) {
                if let Some(node) = self.children.get_mut(c) {
                    node.remove(&data[mismatch..])?;
                }
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
        self.root.insert(data, 0)
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
        assert_eq!(trie.root.children.len(), 2);
        assert_eq!(
            trie.root.children,
            BTreeMap::from_iter([
                (
                    b'a',
                    Node {
                        nelements: 1,
                        content: b"ata".to_vec(),
                        children: BTreeMap::new()
                    }
                ),
                (
                    b'o',
                    Node {
                        nelements: 1,
                        content: b"oto".to_vec(),
                        children: BTreeMap::new()
                    }
                )
            ])
        );
        assert_eq!(trie.insert(b"coco"), (0, 1));
        assert_eq!(trie.insert(b"zaza"), (3, 4));
        assert_eq!(trie.root.nelements, 4);
        assert_eq!(trie.root.content, b"");
        assert_eq!(trie.root.children.len(), 3);
        assert_eq!(
            trie.root.children,
            BTreeMap::from_iter([
                (
                    b'c',
                    Node {
                        nelements: 1,
                        content: b"coco".to_vec(),
                        children: BTreeMap::new()
                    }
                ),
                (
                    b't',
                    Node {
                        nelements: 2,
                        content: b"t".to_vec(),
                        children: BTreeMap::from_iter([
                            (
                                b'a',
                                Node {
                                    nelements: 1,
                                    content: b"ata".to_vec(),
                                    children: BTreeMap::new()
                                }
                            ),
                            (
                                b'o',
                                Node {
                                    nelements: 1,
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
    fn duplicate() {
        let mut trie = Trie::new();
        assert_eq!(trie.insert(b"toto"), (0, 1));
        assert_eq!(trie.insert(b"toto"), (0, 2));
        assert_eq!(trie.root.nelements, 2);
        assert_eq!(trie.remove(b"toto"), Ok(()));
        assert_eq!(trie.root.nelements, 1);
    }
}
