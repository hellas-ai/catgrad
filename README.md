<div style="text-align: center;" align="center">

[![catgrad][catgrad_img]][catgrad_link]

[![Crate][crate_img]][crate_link]
[![License][license_img]][license_file]
[![Documentation][docs_img]][docs_link]

## a categorical deep learning compiler <!-- omit in toc -->

</div>

1. [What is catgrad?](#what-is-catgrad)
1. [Using catgrad](#using-catgrad)
    1. [Defining models](#defining-models)
    1. [Visualising](#visualising-models)
    1. [Typechecking](#typechecking)
    1. [Running models](#running-models)
    1. [Serverless runtime](#serverless-runtime) (coming soon!)
1. [Feature Flags](#feature-flags)
1. [Roadmap](#roadmap)
<!--1. [Category theory](#category-theory)-->

## What is catgrad?

catgrad is a *deep learning **compiler*** using category theory to statically
compile models into their forward and backwards passes.
This means both your inference *and* training loop compile to static code and
run without needing a deep learning framework (not even catgrad!)

A model in catgrad is an [Open Hypergraph](https://docs.rs/open-hypergraphs/)
This means models are completely symbolic mathematical expressions.
Catgrad's autodiff algorithm is a source-to-source transformation of this
symbolic representation, requiring no explicit tracking of backwards passes at
runtime: training and inference are both simply open hypergraphs.

This also means that developing additional backends for catgrad is easy: there
is no need to worry about the backwards pass in Tensor representations, since
it is handled at the *syntactic* level: a catgrad backend is an NdArray
implementation.

We're building Catgrad to power the [Hellas Network](https://hellas.ai/) to
enable trustlessly buying and selling tensor compute in real-time.
The Network uses Catgrad as a serverless runtime: models are serialized and
sent to remote nodes for computation.

## Using Catgrad

There are several ways to use the catgrad rust crate:

1. [Defining models](#defining-models)
1. [Visualising models](#visualising-models)
1. [Typechecking models](#typechecking)
1. [Running models](#running-models)
1. [Serverless runtime](#serverless-runtime) (coming soon!)

An end-to-end example of defining, typechecking, running, and visualising a model
is provided in [./examples/hidden.rs].

### Defining Models

Construct a model or layer by implementing [`crate::stdlib::Def`].
Here's how the standard library defines the `Sigmoid` layer:

```rust,ignore
use catgrad_core::prelude::*;

struct Sigmoid;
impl Def<1, 1> for Sigmoid {
    // Name of the model/definition
    fn path(&self) -> Path {
        path(vec!["nn", "sigmoid"]).unwrap()
    }

    // Definition of Sigmoid as tensor operations
    // This uses the `Var` interface of the Open Hypergraphs library to provide
    // a "rust-like" embedded DSL for building model syntax.
    fn def(&self, graph: &Builder, [x]: [Var; 1]) -> [Var; 1] {
        let c1 = constant_f32(graph, 1.0);
        let s = shape(graph, x.clone());
        let c1 = broadcast(graph, c1, s);

        let r = c1.clone() / (c1 + Exp.call(graph, [-x]));
        [r]
    }

    // Specify the *type* of the model by annotating input/output tensors with
    // symbolic shapes and dtypes.
    fn ty(&self) -> ([Type; 1], [Type; 1]) {
        // API/docs WIP!
    }

};
```

The `def` method constructs the model graph using the
[Var interface](https://docs.rs/open-hypergraphs/latest/open_hypergraphs/lax/var/index.html)
of Open Hypergraphs.

### Visualising models

Use the `svg` feature to visualise models--produce an SVG of the `Sigmoid`
model above with the following code:

```rust,ignore
use catgrad_core::prelude::*;
use catgrad_core::svg::to_svg;

let model = nn::Sigmoid;
let svg_bytes = to_svg(&model.term().unwrap().term);
// then write svg_bytes to a file using std::fs::write.
```

This produces a diagrams of "boxes and wires" as below:

![sigmoid][sigmoid_img]

### Typechecking

Once you've constructed your model, you can typecheck it:

```rust
use catgrad_core::prelude::*;

let model = nn::Sigmoid;
let term = model.term().unwrap(); // extract graph

check::check(&stdlib(), &check::Parameters::from([]), term).expect("typechecking failed");
```

The `stdlib` and `parameters` arguments define what definitions and parameters are in scope when typechecking.

### Running Models

Run a program using an `Interpreter` with a chosen backend.
For example, we can run the Sigmoid layer as follows:

```rust,ignore
use catgrad_core::prelude::*;
use interpreter::{Interpreter, Parameters, Shape, tensor, backend::ndarray::NdArrayBackend};

// choose a backend and get the model as an Open Hypergraph
let backend = NdArrayBackend;
let term = nn::Sigmoid.term().unwrap().term;

// create an input tensor
let input = tensor(&backend, Shape(vec![2, 3]), &[1., 2., 3., 4., 5., 6.]).expect("tensor creation");

// Create and run the interpreter
let interpreter = Interpreter::new(backend, stdlib(), Parameters::from([]));
interpreter.run(term, vec![input]);
```

### Serverless runtime

Coming soon!

## Feature flags

Catgrad core has just one dependency:
[open-hypergraphs](https://docs.rs/open-hypergraphs/),
which itself only relies on `num_traits`.

Optional features can be switched on which add more dependencies:

- `svg` allows producing SVG diagrams of models but requires the `graphviz` and `open-hypergraphs-dot`
- backends:
    - `ndarray-backend` requires the `ndarray` crate
    - `candle-backend` (coming soon!)

## Roadmap

Some features on the roadmap:

- Deterministic and ZK-verified language backends
- Port our autodiff implementation from [python](https://github.com/statusfailed/catgrad)
- Hypergraph Rewriting & optimization [(github issue)](https://github.com/hellas-ai/open-hypergraphs/issues/9)
- Serverless runtime features - package model/weights + run with no env/etc.
- Explicit (in-language) control flow
- Type checker improvements
- Ongoing language changes
- More models

<!--
## Category theory

- Lang & core: categories as "dialects", functors as passes
    - TODO: actually implement this way!
    - Forget as a pass
    - Optimization as rewriting
- Open Hypergraphs and Symmetric Monoidal Categories
- Links to papers
-->

<!-- Badges and Logo -->
[crate_link]: https://crates.io/crates/catgrad "Crate listing"
[crate_img]: https://img.shields.io/crates/v/catgrad.svg?style=for-the-badge&color=f46623 "Crate badge"
[docs_link]: https://docs.rs/catgrad/latest/catgrad "Crate documentation"
[docs_img]: https://img.shields.io/docsrs/catgrad/latest.svg?style=for-the-badge "Documentation badge"
[license_file]: https://github.com/hellas-ai/catgrad/blob/master/LICENSE "Project license"
[license_img]: https://img.shields.io/crates/l/catgrad.svg?style=for-the-badge "License badge"

[catgrad_link]: https://catgrad.com
[catgrad_img]: ./images/catgrad.svg

[sigmoid_img]: ./images/sigmoid.svg
