# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
from __future__ import annotations


class PDETerm:
    """!

    Represents one named, point-wise PDE operator that a solver exposes for
    the user to implement: a C++ method name, its parameter list, and an
    implementation. flux(), eigenvalues(), the nonconservative product,
    algebraic/point sources, boundary conditions, ... all have this same
    shape (see python/exahype2/solvers/fv/riemann/kernels.py for how they
    are actually declared in the generated abstract solver header).

    Well-known terms should get their own PDETerm subclass which pins down
    name and signature and adds any term-specific behaviour - @see Flux for
    a fully worked example. Use the plain PDETerm constructor directly only
    for a new, solver-specific term that does not warrant its own subclass
    (yet).

    A term's implementation is a C++ code snippet that becomes the generated
    method's body verbatim - including the empty string, which generates a
    method that does nothing (a no-op, not "contributes zero": it does not
    zero out the output for you) - or None (the default), which leaves the
    method pure virtual for you to implement by hand in your solver
    subclass instead of generating it. A term you don't want at all is
    simply never passed to solver.add_pde_term() in the first place.

    implementation must not be None when stateless is True: the constructor
    raises ValueError in that case, since a stateless term's generated
    method is static, and static methods cannot be pure virtual - there
    would be no declaration left to signal "implement this by hand".

    PDETerm only models *what* the term is (name, signature, implementation).
    It deliberately knows nothing about Jinja2 or the code generator: turning
    a solver's collection of terms into the generated abstract solver header
    remains the code generator's job, exactly as it is today.

    @see exahype2.api.Solver.add_pde_term()

    """

    NAME: str
    SIGNATURE: str

    RECEIVES_AUXILIARY_VARIABLES: bool = True
    """!
    Whether the generated signature holds pointers to the solver's auxiliary
    variables (@ref exahype2.api.AuxiliaryVariables). False for
    exahype2.api.InitPointWisely.
    """

    WRITES_OWNED_AUXILIARY_VARIABLES: bool = False
    """!
    Whether the term receives writable pointers to the sets which the solver
    owns. True for exahype2.api.InitAuxiliaryVariablesPointWisely.
    """
    
    def __init__(
        self,
        name: str,
        implementation: str = None,
        stateless: bool = False,
    ):
        """!

        name: str
          The C++ method name, e.g. "flux" or "eigenvalues". Used by
          exahype2.api.Solver to key its collection of terms, i.e. has to be
          unique per solver.

        implementation: str
          A C++ snippet that becomes the method's body verbatim (the empty
          string is a valid, generated no-op body), or None (the default) to
          leave the method pure virtual for you to implement by hand. Must
          not be None if stateless is True: the generated method is then
          static, and static methods cannot be pure virtual, so there would
          be nothing left to tell you to implement it by hand.

        """
        if stateless and implementation is None:
            raise ValueError(
                f"PDETerm {name!r} is stateless, so its generated method is static - "
                "static methods cannot be pure virtual, so implementation must not be "
                "None (pass a C++ snippet, or the empty string for a no-op body)."
            )
        self.name = name
        self.implementation = implementation
        self.stateless = stateless

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, implementation={self.implementation!r})"

    def __str__(self) -> str:
        return f"{self.name}: {self.implementation}"

    class Generator:
        """!

        Turns a PDETerm into the C++ text for its generated method: exactly
        one method - the static, GPU-offloadable overload if the term is
        stateless, otherwise the virtual instance method. Never both from
        the same term; @see exahype2.api.Solver.add_pde_term() for how to
        get both variants of the same conceptual operation.

        If term.implementation is None, the method is left for the user to
        implement by hand: pure virtual (" = 0") for the non-stateless case,
        or - since static methods cannot be pure virtual - simply declared
        without a matching out-of-class definition for the stateless case.
        Otherwise the method is generated non-overridable ("final") with
        term.implementation as its body.

        Embedded into PDETerm (reachable as term.generator) so that a term
        subclass can supply its own Generator - @see Flux.Generator - and
        term.generator resolves to the right one automatically, since it is
        looked up on the term's actual class rather than hardcoded to
        PDETerm.Generator.

        """

        def __init__(self, term: "PDETerm", solver=None):
            """!
            @param term: PDETerm
            @param solver: exahype2.api.Solver
              If the solver has auxiliary variables, the generated signature
              contains pointers to them (@ref exahype2.api.AuxiliaryVariables),
              unless the term's RECEIVES_AUXILIARY_VARIABLES is False.
            """
            self.term = term
            self.solver = solver

        @property
        def _has_auxiliary_variables(self) -> bool:
            return (
                self.term.RECEIVES_AUXILIARY_VARIABLES
                and self.solver is not None
                and len(self.solver.auxiliary_variables) > 0
            )

        @property
        def kernel_signature(self) -> str:
            """!
            The signature without the auxiliary variables. This is the
            signature which the compute kernels call.
            """
            return getattr(self.term, "SIGNATURE")

        @property
        def signature(self) -> str:
            """!
            The term's C++ parameter list. Looked up as a SIGNATURE attribute
            on the owning term (e.g. Flux.SIGNATURE); a plain, custom PDETerm
            has to set term.SIGNATURE itself before it can generate code.
            """
            signature = getattr(self.term, "SIGNATURE", None)
            if signature is None:
                raise AttributeError(
                    f"{self.term!r} (or its class) has no SIGNATURE to generate code from; "
                    "either subclass PDETerm and provide one (see Flux), or set "
                    "term.SIGNATURE explicitly."
                )
            if self._has_auxiliary_variables:
                from .AuxiliaryVariables import extend_signature

                return extend_signature(
                    signature,
                    self.solver,
                    self.term.WRITES_OWNED_AUXILIARY_VARIABLES,
                )
            return signature

        def _needs_kernel_adapter(self) -> bool:
            """!
            The compute kernels know nothing about auxiliary variables. They
            call the stateless terms with the original signature. If the
            solver has auxiliary variables, we thus add static overloads with
            the original signature that forward to the term with parameters.
            """
            from .AuxiliaryVariables import split_signature

            return (
                self.term.stateless
                and self._has_auxiliary_variables
                and any(
                    p.split()[-1] in ("Q", "deltaQ")
                    for p in split_signature(self.kernel_signature)
                )
            )

        def _kernel_adapter_arguments(self) -> str:
            from .AuxiliaryVariables import forwarding_arguments

            return ", ".join(forwarding_arguments(self.kernel_signature, self.solver))

        def _generate_kernel_adapter_declaration(self) -> str:
            if not self._needs_kernel_adapter():
                return ""
            return (
                "#if defined(GPUOffloadingOMP)\n"
                "#pragma omp declare target\n"
                "#endif\n"
                f"static GPUCallableMethod void {self.term.name}(\n"
                f"  {self.kernel_signature},\n"
                "  Offloadable\n"
                ");\n"
                "#if defined(GPUOffloadingOMP)\n"
                "#pragma omp end declare target\n"
                "#endif\n"
            )

        def _generate_kernel_adapter_definition(self, qualified_class_name: str) -> str:
            if not self._needs_kernel_adapter():
                return ""
            return (
                "#if defined(GPUOffloadingOMP)\n"
                "#pragma omp declare target\n"
                "#endif\n"
                f"GPUCallableMethod void {qualified_class_name}::{self.term.name}(\n"
                f"  {self.kernel_signature},\n"
                "  Offloadable\n"
                f") {{\n  {self.term.name}({self._kernel_adapter_arguments()}, Offloadable::Yes);\n}}\n"
                "#if defined(GPUOffloadingOMP)\n"
                "#pragma omp end declare target\n"
                "#endif\n"
            )

        def _generate_static_offloadable_declaration(self) -> str:
            """!
            static GPU-offloadable declaration, or the empty string if the
            term isn't stateless. exahype2::Solver::Offloadable is a dummy
            tag argument that exists precisely because C++ cannot overload on
            staticness alone - it is what lets this static method coexist
            with the same-named, non-static, no-Offloadable-argument method
            that registering the same term name again without stateless
            would produce.
            """
            if not self.term.stateless:
                return ""
            return (
                "#if defined(GPUOffloadingOMP)\n"
                "#pragma omp declare target\n"
                "#endif\n"
                f"static GPUCallableMethod void {self.term.name}(\n"
                f"  {self.signature},\n"
                "  Offloadable\n"
                ");\n"
                "#if defined(GPUOffloadingOMP)\n"
                "#pragma omp end declare target\n"
                "#endif\n"
            )

        def generate_declaration(self) -> str:
            """!
            Declaration text for the abstract solver header: the static,
            GPU-offloadable overload if this term is stateless, otherwise
            the virtual instance method - never both from one term. Pure
            virtual (" = 0") if term.implementation is None, "final"
            otherwise; the static, stateless overload is unaffected by this
            distinction (static methods cannot be pure virtual) - @see
            generate_out_of_class_definitions for how it is left for the
            user to implement by hand instead.
            """
            if self.term.stateless:
                return (
                    self._generate_static_offloadable_declaration()
                    + self._generate_kernel_adapter_declaration()
                )
            body_suffix = " = 0" if self.term.implementation is None else " final"
            return (
                f"virtual void {self.term.name}(\n"
                f"  {self.signature}\n"
                f"){body_suffix};\n"
            )

        def generate_definition(self) -> str:
            """!
            Body of the generated method: the term's implementation verbatim.
            Only meaningful when term.implementation is not None - @see
            generate_out_of_class_definitions, which never calls this
            otherwise.
            """
            return self.term.implementation

        def generate_out_of_class_definitions(self, qualified_class_name: str) -> str:
            """!
            Out-of-class method body for the solver's .cpp file: the static,
            GPU-offloadable overload if this term is stateless, otherwise
            the virtual instance method - mirrors generate_declaration(),
            never both from one term. The empty string if
            term.implementation is None: the user implements that method by
            hand instead, whether as an override of the pure virtual
            instance method or, for a stateless term, as an ordinary static
            method matching the declaration (name hiding, not override -
            static methods cannot be virtual).

            qualified_class_name: str
              Fully qualified class name the definition is attached to,
              e.g. "examples::MySolver::AbstractMySolver".
            """
            if self.term.implementation is None:
                return ""
            body = self.generate_definition()
            if self.term.stateless:
                return (
                    "#if defined(GPUOffloadingOMP)\n"
                    "#pragma omp declare target\n"
                    "#endif\n"
                    f"GPUCallableMethod void {qualified_class_name}::{self.term.name}(\n"
                    f"  {self.signature},\n"
                    "  Offloadable\n"
                    f") {{\n  {body}\n}}\n"
                    "#if defined(GPUOffloadingOMP)\n"
                    "#pragma omp end declare target\n"
                    "#endif\n"
                ) + self._generate_kernel_adapter_definition(qualified_class_name)
            return (
                f"void {qualified_class_name}::{self.term.name}(\n"
                f"  {self.signature}\n"
                f") {{\n  {body}\n}}\n"
            )

        def generate_stub_declaration(self) -> str:
            """!
            Override declaration for the concrete solver class that the
            Python API generates when the user hasn't provided their own
            implementation of this term - the empty string if the term
            isn't left abstract (term.implementation is not None; a
            stateless term's method is static and thus never overridable
            in the first place, but that combination cannot occur anyway,
            @see PDETerm.__init__).
            """
            if self.term.implementation is not None or self.term.stateless:
                return ""
            return (
                f"virtual void {self.term.name}(\n"
                f"  {self.signature}\n"
                ") override;\n"
            )

        def generate_stub_definition(self, qualified_class_name: str) -> str:
            """!
            Out-of-class body for generate_stub_declaration(): a trace-in
            (with x, and normal too if the term's signature has one), a
            logWarning noting the method is unimplemented, a
            "// @todo implement" comment, and logTraceOut - or the empty
            string under the same condition as generate_stub_declaration().

            qualified_class_name: str
              Fully qualified class name the definition is attached to,
              e.g. "examples::MySolver::MySolver".
            """
            if self.term.implementation is not None or self.term.stateless:
                return ""
            method_name = f"{self.term.name}(...)"
            log_trace_in = (
                f'logTraceInWith2Arguments("{method_name}", x, normal);'
                if "normal" in self.signature
                else f'logTraceInWith1Argument("{method_name}", x);'
            )
            return (
                f"void {qualified_class_name}::{self.term.name}(\n"
                f"  {self.signature}\n"
                f") {{\n"
                f"  {log_trace_in}\n"
                f'  logWarning("{method_name}", "not implemented yet");\n'
                f"  // @todo implement\n"
                f'  logTraceOut("{method_name}");\n'
                f"}}\n"
            )

    def generator_for(self, solver) -> "PDETerm.Generator":
        """!
        Code generator for this term within a solver, i.e. one that respects
        the solver's auxiliary variables.
        """
        return self.__class__.Generator(self, solver)

    @property
    def generator(self) -> "PDETerm.Generator":
        """!
        Code generator for this term: an instance of this term's own
        (possibly subclass-specific) Generator. @see PDETerm.Generator
        """
        return self.__class__.Generator(self)
