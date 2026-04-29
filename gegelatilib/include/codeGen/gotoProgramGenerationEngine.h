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
#ifndef GOTO_PROGRAM_GENERATION_ENGINE_H
#define GOTO_PROGRAM_GENERATION_ENGINE_H

#include "codeGen/programGenerationEngine.h"

namespace CodeGen {

/**
 * \brief Specialisation of ProgramGenerationEngine for the computed-goto
 *        dispatch style.
 *
 * Differences from the base class:
 *
 *  1. Every program is emitted as a header-only inline function in the
 *     *_program.h file (fileH). The .c file (fileC) is opened but unused,
 *     because ProgramGenerationEngine's constructor always opens both.
 *     The function signature is:
 *
 *       inline __attribute__((always_inline))
 *       fixedpt P<id>(const fixedpt * restrict in1, …, const fixedpt * restrict inN)
 *
 *  2. Data sources map to function parameters ("inN") rather than global
 *     extern pointers.  getNameSourceData() is non-virtually shadowed to
 *     achieve this — the base class method is not virtual.
 *
 *  3. generateProgram() is non-virtually shadowed: it writes the inline
 *     function body directly to fileH rather than fileC.
 *
 *  Note: because ProgramGenerationEngine does not declare its key methods
 *  as virtual, this class shadows (rather than overrides) them. Callers
 *  must hold a GotoProgramGenerationEngine* (or reference) — not a base
 *  pointer — for the correct methods to be dispatched.
 */
class GotoProgramGenerationEngine : public ProgramGenerationEngine
{
  public:
    /**
     * \param filename   Base name; the engine writes to <filename>_program.h.
     * \param env        The GEGELATI Environment (registers, constants, …).
     * \param path       Output directory (trailing '/' required).
     * \param nbInputs   Number of "const fixedpt * restrict inN" parameters
     *                   (default 4).
     */
    GotoProgramGenerationEngine(const std::string& filename,
                                const Environment& env,
                                const std::string& path = "./",
                                int nbInputs = 4);

    /**
     * \brief Destructor — writes the e n d i f (Doxygen is fucking dumb) guard and closes fileH.
     *        fileC is closed by the base destructor (it was opened but never
     *        written to).
     */
    ~GotoProgramGenerationEngine();

    /**
     * \brief Generates one inline program function into fileH.
     *
     * Shadows (does NOT override) ProgramGenerationEngine::generateProgram().
     * Callers must use a GotoProgramGenerationEngine reference.
     *
     * Signature produced:
     *   inline __attribute__((always_inline))
     *   fixedpt P<progID>(const fixedpt * restrict in1, …)
     *
     * \param progID          Unique identifier for the program.
     * \param ignoreException Forwarded to iterateThroughtProgram().
     */
    void generateProgram(uint64_t progID, bool ignoreException);

  protected:
    /**
     * \brief Returns the C name for a data source, mapping inputs to
     *        function parameters "inN" instead of global extern pointers.
     *
     * Shadows (does NOT override) ProgramGenerationEngine::getNameSourceData().
     */
    std::string getNameSourceData(const uint64_t& idx);

  private:
    /// Number of "const fixedpt * restrict inN" parameters.
    int nbInputs;

    /**
     * \brief Opens _program.h, writes include guard + externHeader include.
     *        Also opens (but will not write to) _program.c so that the base
     *        class destructor can close it safely.
     *
     * Called from the constructor after the base class has run so we can
     * replace the header content with the goto-style version.
     *
     * \param filename  Base filename (without extension).
     * \param path      Output directory.
     * \param nbConstant Number of program constants (used for include guard).
     */
    void openGotoFile(const std::string& filename, const std::string& path,
                      size_t nbConstant);
};

} // namespace CodeGen

#endif // GOTO_PROGRAM_GENERATION_ENGINE_H
#endif // CODE_GENERATION