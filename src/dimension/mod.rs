pub mod fixed;
pub mod dynamic;
pub mod axes;
pub mod into;

#[cfg(test)]
mod tests {
    /// Compile-time path check: ensure all four submodules are reachable
    /// via the public dimension module path. If any submodule is renamed
    /// or removed, these `use` statements fail to compile.
    #[test]
    fn test_dimension_submodules_reachable() {
        #[allow(unused_imports)]
        use crate::dimension::fixed;
        #[allow(unused_imports)]
        use crate::dimension::dynamic;
        #[allow(unused_imports)]
        use crate::dimension::axes;
        #[allow(unused_imports)]
        use crate::dimension::into;
    }
}