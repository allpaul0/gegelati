/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2025) :
 *
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

#include "codeGen/tpgGoToGenerationEngine.h"

CodeGen::TPGGoToGenerationEngine::TPGGoToGenerationEngine(
    const std::string& filename, const TPG::TPGGraph& tpg, 
    const std::string& path, bool is_instrumented, bool is_decorated)
    : TPGGenerationEngine(filename, tpg, path, 
        std::make_unique<CodeGen::GotoProgramGenerationEngine>(
        filename + "_" + filenameProg, tpg.getEnvironment(), path, NB_INPUTS))
{
    this->is_instrumented = is_instrumented;
    this->is_decorated = is_decorated;
}

void CodeGen::TPGGoToGenerationEngine::generateTPGGraph()
{
    // Build the ordered vertex list: teams first, then actions.
    orderedVertices.clear();
    auto allVertices = this->tpg.getVertices();

    for (auto* v : allVertices) {
        if (dynamic_cast<const TPG::TPGTeam*>(v) != nullptr) {
            orderedVertices.push_back(v);
        }
    }
    for (auto* v : allVertices) {
        if (dynamic_cast<const TPG::TPGAction*>(v) != nullptr) {
            orderedVertices.push_back(v);
        }
    }

    progGenerationEngine->openFile(progGenerationEngine->filename, 
        progGenerationEngine->path, 
        this->tpg.getEnvironment().getParams().nbProgramConstant);

    initHeaderFile();
    initTpgFile(); // bestProgram() + inferenceTPG() opening + jump_table

    // Emit team blocks then action lines.
    for (auto* v : orderedVertices) {
        if (auto* team = dynamic_cast<const TPG::TPGTeam*>(v)) {
            generateTeam(*team);
        }
    }
    for (auto* v : orderedVertices) {
        if (auto* action = dynamic_cast<const TPG::TPGAction*>(v)) {
            generateAction(*action);
        }
    }

    // Close inferenceTPG().
    fileMain << "}" << std::endl;
}


void CodeGen::TPGGoToGenerationEngine::initTpgFile()
{
    // ---- bestProgram() helper ----
    fileMain
        << "/* ------------------------------------------------------------ */\n"
        << "/* Helper                                                        */\n"
        << "/* ------------------------------------------------------------ */\n"
        << "\n"
        << "static inline int bestProgram(const fixedpt *results, int nb) {\n"
        << "    int   best  = 0;\n"
        << "    fixedpt top = results[0];\n"
        << "    for (int i = 1; i < nb; i++) {\n"
        << "        if (results[i] >= top) { top = results[i]; best = i; }\n"
        << "    }\n"
        << "    return best;\n"
        << "}\n"
        << "\n";

    // ---- inferenceTPG() signature ----
    fileMain
        << "/* ------------------------------------------------------------ */\n"
        << "/* Inference — computed goto dispatch                            */\n"
        << "/* ------------------------------------------------------------ */\n"
        << "\n"
        << "void inferenceTPG(fixedpt *actions";
    for (int i = 1; i <= NB_INPUTS; ++i) {
        fileMain << ",\n                  const fixedpt * restrict in" << i;
    }
    fileMain << ")\n{\n";

    // ---- static jump_table[] ----
    fileMain
        << "    /* Jump table — static const lets GCC keep it in .rodata and\n"
        << "       potentially cache it in a register across iterations.       */\n"
        << "    static const void * const jump_table[] = {\n"
        << "        ";

    for (std::size_t i = 0; i < orderedVertices.size(); ++i) {
        fileMain << "&&L_" << vertexName(*orderedVertices[i]);
        if (i + 1 < orderedVertices.size()) {
            fileMain << ", ";
        }
    }
    fileMain << "\n    };\n\n";

    // ---- initial dispatch to root ----
    const auto& root = *tpg.getRootVertices().at(0);
    fileMain
        << "    /* Initial dispatch — always start at " << vertexName(root) 
        << " */\n"
        << "    goto *jump_table[" << jumpTableIndex(root) << "];"
        << "   /* == &&L_" << vertexName(root) << " */\n"
        << "\n";

    if (is_instrumented) {
        fileMain << "    uint32_t start, end;\n";
    }

    fileMain
        << "    /* ---- Team nodes ----------------------------------------- */\n"
        << "\n";
}

// ============================================================
// initHeaderFile
// ============================================================

void CodeGen::TPGGoToGenerationEngine::initHeaderFile()
{
    // Count teams for the NB_TEAMS macro.
    int nbTeams = 0;
    for (auto* v : this->tpg.getVertices()) {
        if (dynamic_cast<const TPG::TPGTeam*>(v) != nullptr) {
            ++nbTeams;
        }
    }

    fileMainH
        << "#include <stdlib.h>\n"
        << "#include <limits.h>\n"
        << "#include <assert.h>\n"
        << "#include <float.h>\n"
        << "#include <stdbool.h>\n"
        << "#include <stdio.h>\n"
        << "#include <stdint.h>\n"
        << "#include <math.h>\n"
        << "\n"
        << "#include \"externHeader.h\"\n"
        << "\n"
        << "# define NB_TEAMS " << nbTeams << "\n"
        << "\n";

    // inferenceTPG declaration with restrict-qualified parameters.
    fileMainH << "void inferenceTPG(fixedpt* actions";
    for (int i = 1; i <= NB_INPUTS; ++i) {
        fileMainH << ", \n\t\t\t\t\tconst fixedpt * restrict in" << i;
    }
    fileMainH << ");\n" << std::endl;
}

// ============================================================
// generateEdge
// ============================================================

void CodeGen::TPGGoToGenerationEngine::generateEdge(const TPG::TPGEdge& edge)
{
    const Program::Program& p = edge.getProgram();
    uint64_t progID;

    progGenerationEngine->setProgram(p);

    if (findProgramID(p, progID)) {
        progGenerationEngine->generateProgram(progID, false);
    }

    // Emit the call with all input pointers.
    fileMain << "P" << progID << "(";
    for (int i = 1; i <= NB_INPUTS; ++i) {
        fileMain << "in" << i;
        if (i < NB_INPUTS) fileMain << ", ";
    }
    fileMain << ")";
}

// ============================================================
// generateTeam
// ============================================================

void CodeGen::TPGGoToGenerationEngine::generateTeam(const TPG::TPGTeam& team)
{
    auto edges    = team.getOutgoingEdges();
    int  nbEdges  = static_cast<int>(edges.size());
    auto label    = vertexName(team);

    fileMain << "L_" << label << ": {\n";

    // static const int next[] — maps edge index → jump_table index.
    fileMain << "        static const int next[" << nbEdges << "] = { ";
    {
        bool first = true;
        for (auto* edge : edges) {
            if (!first) fileMain << ", ";
            first = false;
            fileMain << jumpTableIndex(*edge->getDestination());
        }
    }
    fileMain << " };\n";

    // scores array.
    fileMain << "        fixedpt  scores[" << nbEdges << "];\n\n";

    // decoration for disassembly code analysis - start
    if (is_decorated) {
        fileMain << "\t";
        fileMain << "__asm__ volatile(\"";
        fileMain << label;
        fileMain << "_start:\");\n";
    }

    // CSR counter READ - start
    if (is_instrumented) {
        fileMain << "\tCSR_READ(CSR_REG_MCYCLE, &start);";
    }

    // One score assignment per edge.
    int i = 0;
    for (auto* edge : edges) {
        fileMain << "        scores[" << i << "] = ";
        generateEdge(*edge);
        fileMain << ";\n";
        ++i;
    }

    // decoration for disassembly code analysis - end
    if (is_decorated) {
        fileMain << "\t";
        fileMain << "__asm__ volatile(\"";
        fileMain << label;
        fileMain << "_end:\");\n";
    }

    // CSR counter READ - end
    if (is_instrumented) {
        fileMain << "\tCSR_READ(CSR_REG_MCYCLE, &end);";
    }

    // Dispatch.
    fileMain << "\n"
             << "        goto *jump_table[next[bestProgram(scores, "
             << nbEdges << ")]];\n"
             << "    }\n\n";
}

// ============================================================
// generateAction
// ============================================================

void CodeGen::TPGGoToGenerationEngine::generateAction(
    const TPG::TPGAction& action)
{
    uint64_t id = action.getActionID();
    fileMain << "L_A" << id << ": actions[0] = " << id << "; return;\n";
}

// ============================================================
// vertexName
// ============================================================

std::string CodeGen::TPGGoToGenerationEngine::vertexName(
    const TPG::TPGVertex& v)
{
    std::ostringstream oss;
    if (dynamic_cast<const TPG::TPGTeam*>(&v) != nullptr) {
        oss << "T" << findVertexID(v);
    }
    else {
        oss << "A" << dynamic_cast<const TPG::TPGAction*>(&v)->getActionID();
    }
    return oss.str();
}

// ============================================================
// jumpTableIndex
// ============================================================

int CodeGen::TPGGoToGenerationEngine::jumpTableIndex(
    const TPG::TPGVertex& v) const
{
    for (int i = 0; i < static_cast<int>(orderedVertices.size()); ++i) {
        if (orderedVertices[i] == &v) {
            return i;
        }
    }
    throw std::runtime_error(
        "TPGGoToGenerationEngine::jumpTableIndex: vertex not found");
}

#endif // CODE_GENERATION