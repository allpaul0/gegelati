/**
 * GEGELATI is an open-source reinforcement learning framework for training
 * artificial intelligence based on Tangled Program Graphs (TPGs).
 *
 * This software is governed by the CeCILL-C license under French law and
 * abiding by the rules of distribution of free software. You can use,
 * modify and/ or redistribute the software under the terms of the CeCILL-C
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 *
 * As a counterpart to the access to the source code and rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty and the software's author, the holder of the
 * economic rights, and the successive licensors have only limited
 * liability.
 *
 * In this respect, the user's attention is drawn to the risks associated
 * with loading, using, modifying and/or developing or reproducing the
 * software by the user in light of its specific status of free software,
 * that may mean that it is complicated to manipulate, and that also
 * therefore means that it is reserved for developers and experienced
 * professionals having in-depth computer knowledge. Users are therefore
 * encouraged to load and test the software's suitability as regards their
 * requirements in conditions enabling the security of their systems and/or
 * data to be ensured and, more generally, to use and operate it in the
 * same conditions as regards security.
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL-C license and that you accept its terms.
 */

#ifdef CODE_GENERATION

#include "codeGen/gotoProgramGenerationEngine.h"
#include "util/timestamp.h"

// ============================================================
// Constructor
// ============================================================

CodeGen::GotoProgramGenerationEngine::GotoProgramGenerationEngine(
    const std::string& filename, const Environment& env,
    const std::string& path, int nbInputs)
    : ProgramGenerationEngine(filename, env, path), nbInputs(nbInputs)
{
    // The base class constructor has already called openFile(), which:
    //   - opened fileC  ( filename _program.c)  — we will not write to it
    //   - opened fileH  ( filename _program.h)  — we rewrite it below
    //   - wrote global extern declarations into fileC
    //   - wrote include guards + externHeader into fileH
    //
    // We close fileH and reopen it with our goto-style header content.
    // fileC is left open (but empty of useful content); the base destructor
    // will close it.
    fileH.close();
    openGotoFile(filename, path, env.getParams().nbProgramConstant);
}

// ============================================================
// Destructor
// ============================================================

CodeGen::GotoProgramGenerationEngine::~GotoProgramGenerationEngine()
{
    if (fileH.is_open()) {
        fileH << "\n#endif // GEGELATI_GENERATED\n" << std::flush;
        fileH.close();
        // Tell the base class not to emit another #endif.
        this->headerClosed = true;
    }
}

// ============================================================
// openGotoFile
// ============================================================

void CodeGen::GotoProgramGenerationEngine::openGotoFile(
    const std::string& filename, const std::string& path,
    size_t /*nbConstant*/)
{
    // Re-open fileH with goto-style content (header-only; no .c needed).
    fileH.open(path + filename + ".h", std::ofstream::out);
    if (!fileH.is_open()) {
        throw std::runtime_error(
            "GotoProgramGenerationEngine: cannot open " + path + filename +
            ".h");
    }

    fileH << "/**\n"
          << " * File generated with GEGELATI v" GEGELATI_VERSION "\n"
          << " * On the " << Util::getCurrentDate() << "\n"
          << " * With the " << DEMANGLE_TYPEID_NAME(typeid(*this).name())
          << ".\n"
          << " */\n";

    fileH << "#ifndef C_" << filename << "_H\n"
          << "#define C_" << filename << "_H\n"
          << "\n"
          << "#include \"externHeader.h\"\n"
          << "\n";

    fileH.flush(); // ensure header prologue is on disk before any redirections

}

// ============================================================
// generateProgram  (shadows, does not override, the base method)
// ============================================================

void CodeGen::GotoProgramGenerationEngine::generateProgram(uint64_t progID,
                                                           bool ignoreException)
{
    // Build the comma-separated parameter list once.
    std::ostringstream params;
    for (int i = 1; i <= nbInputs; ++i) {
        if (i > 1) params << ", ";
        params << "const fixedpt * restrict in" << i;
    }

    // Emit the inline function signature directly into fileH.
    fileH << "\ninline __attribute__((always_inline)) fixedpt P" << progID
          << "(" << params.str() << ") {\n";

    // -- Register array --
    int nbReg = static_cast<int>(
        this->program->getEnvironment().getParams().nbRegisters);
    fileH << "\tfixedpt reg[" << nbReg << "] = {";
    for (int i = 0; i < nbReg; ++i) {
        fileH << "0";
        if (i < nbReg - 1) fileH << ", ";
    }
    fileH << "};\n";

    // -- Constants array (if any) --
    int nbCst = static_cast<int>(
        this->program->getEnvironment().getParams().nbProgramConstant);
    if (nbCst > 0) {
        fileH << "\tint32_t " << nameConstantVariable << "[" << nbCst
              << "] = {";
        for (int i = 0; i < nbCst; ++i) {
            fileH << this->program->getConstantAt(i).value;
            if (i < nbCst - 1) fileH << ", ";
        }
        fileH << "};\n";
    }

    // flush both streams then redirect fileC's buffer to fileH's buffer so the
    // output lands in the header.  Flushing prevents interleaving/corruption.
    fileH.flush();
    fileC.flush();

    // -- Program lines --
    // generateCurrentLine() writes to fileC.  We redirect fileC's underlying
    // buffer to fileH's buffer so the output lands in the header.
    // std::ofstream inherits rdbuf(streambuf*) from std::ostream, so we cast
    // to the base to access the two-argument form.
    std::ostream& cAsOstream  = static_cast<std::ostream&>(fileC);
    std::streambuf* savedCBuf = cAsOstream.rdbuf(fileH.rdbuf());

    iterateThroughtProgram(ignoreException);

    // Restore fileC's buffer (even though we never actually write to it).
    cAsOstream.rdbuf(savedCBuf);
    fileC.flush();
    fileH.flush();

    // -- Return --
    fileH << "\treturn reg[0];\n}\n";
}

// ============================================================
// getNameSourceData  (shadows the base method)
// ============================================================

std::string CodeGen::GotoProgramGenerationEngine::getNameSourceData(
    const uint64_t& idx)
{
    // idx 0          → registers  ("reg")
    // idx 1          → constants  ("cst"), only if nbProgramConstant > 0
    // idx 1 (or 2)…  → input arrays: "in1", "in2", …

    if (idx == 0) {
        return nameRegVariable; // "reg"
    }

    if (this->program->getEnvironment().getParams().nbProgramConstant > 0 &&
        idx == 1) {
        return nameConstantVariable; // "cst"
    }

    // Compute the 1-based "inN" index.
    // Without constants: idx 1 → in1, idx 2 → in2, …
    // With constants:    idx 2 → in1, idx 3 → in2, …
    uint64_t inputIdx = idx;
    if (this->program->getEnvironment().getParams().nbProgramConstant > 0) {
        inputIdx--; // shift past the constants slot
    }
    return nameDataVariable + std::to_string(inputIdx); // "in1", "in2", …
}

#endif // CODE_GENERATION