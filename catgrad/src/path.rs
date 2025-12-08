//! Names of operations in surface syntax
use serde::{Deserialize, Serialize};
use std::fmt;
use std::slice;

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct Path(Vec<PathComponent>);

// Names of definitions
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct PathComponent(String); // only [a-zA-Z_]

pub fn path(components: Vec<&str>) -> Result<Path, InvalidPathComponent> {
    Path::empty().extend(components)
}

impl Path {
    pub fn iter(&self) -> slice::Iter<'_, PathComponent> {
        self.0.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn new<T: AsRef<str>, I: IntoIterator<Item = T>>(
        x: I,
    ) -> Result<Path, InvalidPathComponent> {
        Ok(Path(
            x.into_iter()
                .map(|x| PathComponent::try_from(x.as_ref()))
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }

    pub fn empty() -> Self {
        Path(vec![])
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn concat(&self, other: &Path) -> Path {
        let mut components = self.0.clone();
        components.extend(other.0.clone());
        Path(components)
    }

    pub fn extend<T: AsRef<str>, I: IntoIterator<Item = T>>(
        &self,
        components: I,
    ) -> Result<Path, InvalidPathComponent> {
        let p = Path::new(components)?;
        Ok(self.concat(&p))
    }
}

////////////////////////////////////////////////////////////////////////////////
// Display/TryFrom instances

#[derive(Debug, Clone, PartialEq)]
pub struct InvalidPathComponent(pub String);

impl TryFrom<String> for PathComponent {
    type Error = InvalidPathComponent;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value.chars().all(|c| c.is_alphanumeric() || c == '_') {
            Ok(PathComponent(value))
        } else {
            Err(InvalidPathComponent(value))
        }
    }
}

impl TryFrom<&str> for PathComponent {
    type Error = InvalidPathComponent;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        value.to_string().try_into()
    }
}

impl fmt::Display for PathComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Display for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let components: Vec<String> = self.0.iter().map(|c| c.to_string()).collect();
        write!(f, "{}", components.join("."))
    }
}
