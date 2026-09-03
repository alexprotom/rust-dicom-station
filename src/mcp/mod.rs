//! The MCP server behind `rds-mcp`: what an agent may ask the station to do,
//! and the safety layer every answer passes through.
//!
//! Layout:
//!
//! * [`config`] - the operator's `mcp.toml`: roots, output folder, PHI
//!   policy. Nothing in it can be changed through the protocol.
//! * [`phi`] - the gate (which datasets still name their patient) and the
//!   door (the [`phi::Redactor`] every outgoing string passes).
//! * [`session`] - the open datasets, transforms and reports, with the
//!   handles the client refers to them by.
//! * [`tools`] - the tools themselves, as plain functions on a [`Core`].
//! * [`prompts`] - the `heart_target_propagation` prompt and the documents
//!   served as resources.
//! * [`audit`] - the call log.
//! * [`server`] - the `rmcp` glue: transport, progress, cancellation.
//!
//! The tools are written against [`Core::call`], a synchronous function
//! taking a name and JSON arguments, so that the same code runs under the
//! protocol, in an in-process test, and one day inside the viewer. The
//! `rmcp` layer is deliberately thin.
//!
//! Compiled only with the `mcp` feature; the viewer never links any of it.

pub mod audit;
pub mod config;
pub mod phi;
pub mod prompts;
pub mod server;
pub mod session;
pub mod tools;

use anyhow::Result;
use serde_json::Value;

use crate::progress::Progress;

pub use config::Config;
pub use session::Session;

/// One tool as the client sees it.
pub struct ToolSpec {
    pub name: &'static str,
    pub description: &'static str,
    /// JSON schema of the arguments object.
    pub schema: Value,
    /// Whether the call may run for more than a few seconds, so an
    /// `_async` twin is offered.
    pub long: bool,
}

/// The server's state plus the dispatch. Everything the protocol layer does
/// is `list` and `call`.
pub struct Core {
    pub session: Session,
}

impl Core {
    pub fn new(config: Config) -> Core {
        Core {
            session: Session::new(config),
        }
    }

    /// Run one tool. The result is *not* yet redacted: [`Core::call`] is
    /// the arithmetic, [`Core::call_public`] is the door.
    fn call_raw(&mut self, name: &str, args: Value, p: &Progress) -> Result<Value> {
        tools::dispatch(self, name, args, p)
    }

    /// Run one tool and pass its answer (or its error) through the
    /// redactor. This is the only entry point the server and the tests use.
    pub fn call_public(
        &mut self,
        name: &str,
        args: Value,
        p: &Progress,
    ) -> Result<Value, phi::Public> {
        match self.call_raw(name, args, p) {
            Ok(v) => Ok(self.session.json(v)),
            Err(e) => Err(self.session.text(&format!("{e:#}"))),
        }
    }
}

/// Every tool, in the order the client lists them.
pub fn tool_specs() -> Vec<ToolSpec> {
    tools::specs()
}
