/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2026) :
 * Paul Allaire <paul.allaire@insa-rennes.fr> (2026)
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
#ifndef TPG_GOTO_GENERATION_ENGINE_H
#define TPG_GOTO_GENERATION_ENGINE_H

#include "codeGen/gotoProgramGenerationEngine.h"
#include "codeGen/tpgGenerationEngine.h"

namespace CodeGen {

/**
 * \brief Generation engine producing a TPG inference function that uses
 *        GCC computed-goto dispatch (&&label / goto *ptr)
 *
 * Generated TPG.c structure:
 *  - static inline bestProgram() helper,
 *  - inferenceTPG(fixedpt *actions, const fixedpt * restrict in1, …) body,
 *  - a static const jump_table[] of &&label addresses,
 *  - per-team  L_T<id>: { … goto *jump_table[next[best]]; } blocks,
 *  - per-action L_A<id>: actions[0] = <id>; return;  lines.
 *
 * Generated _program.h: all programs as
 *   inline __attribute__((always_inline)) fixedpt P<id>(const fixedpt * restrict in1, …)
 *
 * Design note — two program engines:
 *   TPGGenerationEngine has a ProgramGenerationEngine attribute but when a
 *   TPGGoToGenerationEngine is used, that attribute is a GotoProgramGenerationEngine.
 * 
 *   This allows us to reuse the base class's program generation logic while
 *   overriding the program generation style in the GotoProgramGenerationEngine.
 */
class TPGGoToGenerationEngine : public TPGGenerationEngine
{
  public:
    /**
     * \brief Constructor.
     *
     * \param filename  Base name for the generated .c/.h files.
     * \param tpg       The TPG graph to generate code for.
     * \param path      Output directory (trailing '/' required).
     * \param is_instrumented denotes instrumentation at team level
     * \param is_decorated denotes decoration at team level for disassembly inspection
     */
    TPGGoToGenerationEngine(const std::string& filename,
                            const TPG::TPGGraph& tpg,
                            const std::string& path = "./",
                            bool is_instrumented = false,
                            bool is_decorated = false);

    /**
     * \brief Destructor — base class handles closing fileMain/fileMainH.
     */
    ~TPGGoToGenerationEngine() = default;

    /**
     * \brief Top-level entry point: generates the complete TPG .c/.h pair.
     */
    void generateTPGGraph() override;

  protected:
    /**
     * \brief Emits  scores[i] = P<id>(in1,…,in4);  into fileMain and
     *        triggers inline program generation into _program.h via gotoProg.
     */
    void generateEdge(const TPG::TPGEdge& edge) override;

    /**
     * \brief Emits the L_T<id>: { static next[]; scores[]; … dispatch } block.
     */
    void generateTeam(const TPG::TPGTeam& team) override;

    /**
     * \brief Emits  L_A<id>: actions[0] = <id>; return;
     */
    void generateAction(const TPG::TPGAction& action) override;

    /**
     * \brief Returns "T<vertexID>" for teams, "A<actionID>" for actions.
     *
     * Not declared with 'override' — vertexName is not part of the
     * TPGGenerationEngine virtual interface.
     */
    std::string vertexName(const TPG::TPGVertex& v);

  private:
    /**
     * \brief Writes bestProgram() helper and opens inferenceTPG() with the
     *        jump_table[] and initial goto into fileMain.
     */
    void initTpgFile() override;

    /**
     * \brief Writes #includes, NB_TEAMS macro, and inferenceTPG declaration
     *        into fileMainH.
     */
    void initHeaderFile() override;

    /**
     * \brief Returns the jump_table[] index for a vertex.
     *
     * Teams are ordered first (indices 0…nbTeams-1), then actions.
     */
    int jumpTableIndex(const TPG::TPGVertex& v) const;

    /// Ordered vertex list (teams first, then actions) built in generateTPGGraph().
    std::vector<const TPG::TPGVertex*> orderedVertices;

    /// Number of input-array parameters passed to every program.
    static constexpr int NB_INPUTS = 4;

    /// boolean set if we instrument the TPG at inference at team level
    /// all programs of a given team are surrounded by CSR reads 
    bool is_instrumented;

    /// boolean set if we decorate the TPG at inference at team level
    /// all programs of a given team are surrounded by additional assembly 
    /// start & end labels 
    bool is_decorated;
};

} // namespace CodeGen

#endif // TPG_GOTO_GENERATION_ENGINE_H
#endif // CODE_GENERATION