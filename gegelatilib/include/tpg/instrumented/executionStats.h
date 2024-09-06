/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2022) :
 *
 * Elinor Montmasson <elinor.montmasson@gmail.com> (2022)
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

#ifndef EXECUTION_STATS_H
#define EXECUTION_STATS_H

#include <map>

#include "tpg/instrumented/tpgExecutionEngineInstrumented.h"
#include "tpg/tpgGraph.h"

namespace TPG {

    // inference trace is a keyword for visited vertices from one inference.
    // As the inference of a TPG varies according to its execution context,
    // inference traces are the history of the execution path of a TPG from
    // the starting root/Team, passing by every visited vertices/Team to the terminal 
    // leaf/action. 

    /**
     * \brief Store execution statistics of one inference trace.
     *
     * It contains :
     * - the inference trace in a std::vector<const TPG::TPGVertex*>
     * - the number of evaluated teams
     * - the number of evaluated programs
     * - the number of executed lines
     * - the number of execution for each instructions (indexed by instruction
     * index)
     */
    struct InferenceTraceStats
    {
        /// The inference trace.
        const std::vector<const TPG::TPGVertex*> inferenceTrace;

        /// Number of team evaluated.
        const uint64_t nbEvaluatedTeams;
        /// Number of programs evaluated.
        const uint64_t nbEvaluatedPrograms;
        /// Number of program lines executed.
        const uint64_t nbExecutedLines;
        /// Map that associate the instruction indexes with the number of
        /// execution of the corresponding Instruction.
        const std::map<uint64_t, uint64_t> nbExecutionForEachInstr;
    };

    /**
     * \brief Utility class for extracting execution statistics
     * from a TPGExecutionEngineInstrumented and an instrumented TPGGraph.
     *
     * The main method of this class is analyzeExecution(), which will :
     *  - retrieve from a TPGGraph the instrumented values and compute
     *  average execution statistics.
     *  - compute execution statistics for every inference done with a
     * TPGExecutionEngineInstrumented.
     *  - create distributions from statistics of analyzed inference traces.
     *
     * Before analyzing or even starting any inference, you must :
     *  - use a TPGGraph associated to a TPGFactoryInstrumented.
     *  - use a TPGExecutionEngineInstrumented that will execute the TPGGraph.
     *  - clear any previous instrumented data :
     *      --> for the TPGGraph, use
     * TPGFactoryInstrumented::resetTPGGraphCounters().
     *      --> for the TPGExecutionEngineInstrumented, use its method
     * TPGExecutionEngineInstrumented::clearInferenceTraceHistory(). Otherwise, the
     * results won't have any meaning. If you have never executed the TPGGraph
     * or the TPGExecutionEngineInstrumented, resetting them isn't required.
     *
     * You can then execute the TPG for as many inferences as you like.
     *
     * Then, use analyzeExecution() with the TPGGraph and
     * TPGExecutionEngineInstrumented, and access the statistics using the
     * provided getters and setters.
     *
     * Warning : the class deduces the number of inferences based on the sum
     * of evaluation each root vertices had. If you executed your TPGGraph
     * starting from multiple roots, then remember that the average statistics
     * are based on ALL inferences, regardless of the root vertices used.
     *
     * The Json exporter is designed to be used after a call to
     * analyzeExecution(). Just call writeStatsToJson() to export statistics in
     * a file with json format.
     */
    class ExecutionStats
    {

      private:
        /* Average results */

        /// Average number of evaluated teams per inference.
        double avgNbEvaluatedTeamsPerInf = 0.0;

        /// Average number of programs evaluated per inference.
        double avgNbEvaluatedProgramsPerInf = 0.0;

        /// Average number of executed lines per inference.
        double avgNbExecutedLinesPerInf = 0.0;

        /**
         * This map associate an Instruction identifier from an
         * instruction set with the average number of execution
         * of the instruction per inference.
         */
        std::map<size_t, double> avgNbExecutionForEachInstrPerInf;

        /* Analyzed inference traces */

        /// Statistics of last analyzed inference traces.
        std::vector<InferenceTraceStats> vecInferenceTraceStats;

        /* Distributions */

        /**
         * \brief Distribution of the number of evaluated team per inference for
         * all analyzed inference traces.
         *
         * distribNbEvaluatedTeamsPerInf[x] = y --> y inferences evaluated x teams.
         */
        std::map<size_t, size_t> distribNbEvaluatedTeamsPerInf;

        /**
         * \brief Distribution of the number of evaluated programs per inference
         * for all analyzed inference traces.
         *
         * distribNbEvaluatedProgramsPerInf[x] = y --> y inferences evaluated x
         * programs.
         */
        std::map<size_t, size_t> distribNbEvaluatedProgramsPerInf;

        /**
         * \brief Distribution of the number of executed lines per inference for
         * all analyzed inference traces.
         *
         * distribNbExecutedLinesPerInf[x] = y --> y inferences executed x lines.
         */
        std::map<size_t, size_t> distribNbExecutedLinesPerInf;

        /**
         * \brief Distributions of the number of executions of each instruction
         * per inference for all analyzed inference traces.
         *
         * distribNbExecutionForEachInstrPerInf[i][x] = y --> for instruction at
         * index i, y inferences executed this instruction x times.
         */
        std::map<size_t, std::map<size_t, size_t>> distribNbExecutionForEachInstrPerInf;

        /**
         * \brief Distribution of the number of visit each vertex had for all
         * analyzed inference traces.
         *
         * distribNbVisitForEachVertexPerInf[v] = y --> y inferences visited vertex pointed by
         * v.
         */
        std::map<const TPG::TPGVertex*, size_t> distribNbVisitForEachVertexPerInf;

        /// Graph used during last call to analyzeExecution
        const TPGGraph* lastAnalyzedGraph = nullptr;

      protected:
        /**
         * \brief Analyze a program to get how many times each instruction is
         * used.
         *
         * \param[out] nbExecutionForEachInstr the std::map<uint64_t, uint64_t>& that
         * will be incremented for each instruction use.
         * \param[in] program the analyzed program.
         */
        static void analyzeProgram(
            std::map<uint64_t, uint64_t>& nbExecutionForEachInstr,
            const Program::Program& program);

      public:
        /// Default constructor.
        ExecutionStats() = default;

        /// Default destructor.
        virtual ~ExecutionStats() = default;

        /**
         * \brief Analyze the average statistics of an instrumented TPGGraph
         * execution.
         *
         * Results are stored in the average results class attributes.
         *
         * \param[in] graph the analyzed TPGGraph*.
         * \throws std::bad_cast if graph contains at least one non instrumented
         * vertex or edge.
         */
        void analyzeInstrumentedGraph(const TPGGraph* graph);

        /**
         * \brief Analyze the execution statistics of one inference trace.
         *
         * The vector inferenceTrace contains all visited vertices for one inference
         * in order : inferenceTrace[0] is the root, and inferenceTrace[inferenceTrace.size()-1] the
         * action.
         *
         * Results are stored in a new inferenceTraceStats struct which is pushed back
         * in attribute vecInferenceTraceStats. Previous results will be erased.
         *
         * \param[in] inferenceTrace a vector<const TPGVertex*> of the analyzed inference
         * trace.
         */
        void analyzeInferenceTrace(const std::vector<const TPGVertex*>& inferenceTrace);

        /**
         * \brief Analyze the execution statistics of multiple inferences
         * done with a TPGExecutionEngineInstrumented.
         *
         * Previous results will be erased.
         *
         * \param[in] tee the TPGExecutionEngineInstrumented.
         * \param[in] graph the TPGGraph executed with tee.
         * \throws std::bad_cast if the graph contains a non instrumented vertex
         * or edge.
         */
        void analyzeExecution(const TPG::TPGExecutionEngineInstrumented& tee,
                              const TPGGraph* graph);

        /// Get the average number of evaluated teams per inference.
        double getAvgNbEvaluatedTeamsPerInf() const;

        /// Get the average number of programs evaluated per inference.
        double getAvgNbEvaluatedProgramsPerInf() const;

        /// Get the average number of executed lines per inference.
        double getAvgNbExecutedLinesPerInf() const;

        /// Get a reference to the map that associate each instruction to
        /// its average number of execution per inference.
        const std::map<size_t, double>& getAvgNbExecutionForEachInstrPerInf() const;

        /// Get stored inferenceTrace statistics.
        const std::vector<InferenceTraceStats>& getInferenceTracesStats() const;

        /// Get the distribution of the number of evaluated teams.
        const std::map<size_t, size_t>& getDistribNbEvaluatedTeamsPerInf() const;

        /// Get the distribution of the number of evaluated programs.
        const std::map<size_t, size_t>& getDistribNbEvaluatedProgramsPerInf() const;

        /// Get the distribution of the number of executed lines.
        const std::map<size_t, size_t>& getDistribNbExecutedLinesPerInf() const;

        /// Get distributions of the number of executions for each instruction.
        const std::map<size_t, std::map<size_t, size_t>>&
        getDistribNbExecutionForEachInstrPerInf() const;

        /// Get the distribution if the number of visit for each vertex.
        const std::map<const TPG::TPGVertex*, size_t>& getDistribNbVisitForEachVertexPerInf() const;

        /// Clear stored inferneceTrace statistics and distributions.
        void clearInferenceTracesStats();

        /**
         * \brief Export the execution statistics of the last analyzeExecution()
         * call to a file using Json format.
         *
         * This method will use the statistics currently stored in the object
         * because it is intended to be used after a call to analyzeExecution().
         * Using it after separate calls to analyzeInstrumentedGraph() or
         * analyzeInferenceTrace() might lead to uncorrelated data.
         *
         * Data is organized as follows :
         *
         *      {
         *          "ExecutionStats" :
         *          {
         *              "avgNbEvaluatedTeamsPerInf" : value,
         *              "avgNbEvaluatedProgramsPerInf" : value,
         *              "avgNbExecutedLinesPerInf" : value,
         *              "avgNbExecutionForEachInstrPerInf" :
         *              {
         *                  "InstructionIndex" : nbExecution,
         *                  ...
         *              },
         *              "distributionNbEvaluatedProgramsPerInf" :
         *              {
         *                  "N" : count of inferences which evaluated N
         * programs,
         *                  ...
         *              },
         *              "distributionNbEvaluatedTeamsPerInf" :
         *              {
         *                  "N" : count of inferences which evaluated N teams,
         *                  ...
         *              },
         *              "distributionNbExecutedLinesPerInf" :
         *              {
         *                  "N" : count of inferences which executed N lines,
         *                  ...
         *              },
         *              "distribNbExecutionForEachInstrPerInf" :
         *              {
         *                  "InstructionIndex" :
         *                  {
         *                      "N" : count of inferences which executed the
         * instruction N times,
         *                      ...
         *                  },
         *                  ...
         *              },
         *              "distributionNbVisitForEachVertexPerInf" :
         *              {
         *                  "VertexIndex" : count of inferences which visited
         * the vertex,
         *                  ...
         *              }
         *          },
         *
         *          "vecInferenceTraceStats" :
         *          {
         *              "inferenceTraceNumber" :
         *              {
         *                  "nbEvaluatedPrograms" : value,
         *                  "nbEvaluatedTeams" : value,
         *                  "nbExecutedLines" : value,
         *                  "nbExecutionPerInstruction" :
         *                  {
         *                      "InstructionIndex" : nbExecution,
         *                  ...
         *                  },
         *                  "inferenceTrace" : [Array of vertices indexes in the TPGGraph]
         *              },
         *              ...
         *          }
         *
         *      }
         *
         * \param[in] filePath the path to the output file.
         * \param[in] noIndent true if the json format must not be indented.
         * Files can become large quickly with a lot of inferenceTraces, so if the file
         * will just be analyzed by another program, set this to true to save
         * some space on your disk. Set noIndent to false if you want to keep
         * the file readable.
         */
        void writeStatsToJson(const char* filePath, bool noIndent = false) const;
    };

} // namespace TPG

#endif // EXECUTION_STATS_H
