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

#include "tpg/instrumented/executionStats.h"

#include <algorithm>
#include <fstream>
#include <json.h>
#include <numeric>
#include <vector>

#include "program/program.h"
#include "tpg/instrumented/tpgActionInstrumented.h"
#include "tpg/instrumented/tpgEdgeInstrumented.h"
#include "tpg/instrumented/tpgTeamInstrumented.h"
#include "tpg/instrumented/tpgVertexInstrumented.h"

void TPG::ExecutionStats::analyzeProgram(
    std::map<uint64_t, uint64_t>& instructionCounts,
    const Program::Program& program)
{

    for (int i = 0; i < program.getNbLines(); i++) {
        instructionCounts[program.getLine(i).getInstructionIndex()]++;
    }
}

void TPG::ExecutionStats::analyzeInstrumentedGraph(const TPGGraph* graph)
{
    this->avgNbExecutionPerInstruction.clear();

    const std::vector<const TPG::TPGVertex *> roots = graph->getRootVertices();
    uint64_t nbInferences = std::accumulate(roots.cbegin(), roots.cend(), (uint64_t)0,
        [](uint64_t accu, const TPGVertex* vertex) {
            // Raise std::bad_cast if not an instrumented team
            const TPG::TPGTeamInstrumented& rootTeam = dynamic_cast<const TPGTeamInstrumented&>(*vertex);
            
            return accu + rootTeam.getNbVisits();
        });

    uint64_t nbEvaluatedTeams = 0;
    uint64_t nbEvaluatedPrograms = 0;
    uint64_t nbExecutedLines = 0;
    std::map<size_t, uint64_t> totalExecutionsPerInstruction;

    const std::vector<const TPG::TPGVertex *> vertices = graph->getVertices();

    for (const TPG::TPGVertex* vertex : vertices) {

        // Skip non-team instrumented vertices
        if (dynamic_cast<const TPGActionInstrumented*>(vertex)) 
            continue;

        // Raise std::bad_cast if not an instrumented team
        const TPG::TPGTeamInstrumented* team = dynamic_cast<const TPGTeamInstrumented*>(vertex);

        nbEvaluatedTeams += team->getNbVisits();

        for (const TPG::TPGEdge *edge : team->getOutgoingEdges()) {

            // Raise std::bad_cast if not an instrumented edge
            const TPGEdgeInstrumented& tpgEdgeInstrumented = dynamic_cast<const TPGEdgeInstrumented&>(*edge);

            uint64_t nbEdgeEval = tpgEdgeInstrumented.getNbVisits();
            const Program::Program& edgeProgram = tpgEdgeInstrumented.getProgram();

            // Evaluated edge => edge program executed
            if (nbEdgeEval > 0) {
                nbEvaluatedPrograms += nbEdgeEval;
                nbExecutedLines += nbEdgeEval * edgeProgram.getNbLines();

                std::map<uint64_t, uint64_t> linesPerInstruction;
                analyzeProgram(linesPerInstruction, edgeProgram);
                for (const std::pair<const size_t, const size_t>& pair : linesPerInstruction) {
                    totalExecutionsPerInstruction[pair.first] += nbEdgeEval * pair.second;
                }
            }
        }
    }

    this->avgEvaluatedTeams = (double)nbEvaluatedTeams / (double)nbInferences;
    this->avgEvaluatedPrograms = (double)nbEvaluatedPrograms / (double)nbInferences;
    this->avgExecutedLines = (double)nbExecutedLines / (double)nbInferences;

    for (const auto& p : totalExecutionsPerInstruction) {
        avgNbExecutionPerInstruction[p.first] = (double)p.second / (double)nbInferences;
    }
}

void TPG::ExecutionStats::analyzeInferenceTrace(const std::vector<const TPGVertex*>& inferenceTrace)
{
    // Do not count the action vertex at the end
    uint64_t nbEvaluatedTeams = inferenceTrace.size() - 1;
    uint64_t nbEvaluatedPrograms = 0;
    uint64_t nbExecutedLines = 0;
    std::map<uint64_t, uint64_t> nbExecutionPerInstruction;

    std::cout << "Traces" << std::endl;

    for (std::vector<const TPG::TPGVertex *>::const_iterator inferenceTraceTeamsIterator = inferenceTrace.cbegin(); inferenceTraceTeamsIterator != inferenceTrace.cend() - 1; inferenceTraceTeamsIterator++) {
        std::cout << *inferenceTraceTeamsIterator << std::endl;
    }
    
    std::cout << std::endl;

    // For of each visited teams, analysing its edges
    for (std::vector<const TPG::TPGVertex *>::const_iterator inferenceTraceTeamsIterator = inferenceTrace.cbegin(); inferenceTraceTeamsIterator != inferenceTrace.cend() - 1; inferenceTraceTeamsIterator++) {
        for (const TPG::TPGEdge* edge : (*inferenceTraceTeamsIterator)->getOutgoingEdges()) {

            std::cout << "edge->getDestination(): " << edge->getDestination() << std::endl;
                 
            // Edges leading to a previously visited teams (including the current team) are not evaluated
            auto endSearchIt = inferenceTraceTeamsIterator + 1;

            std::cout << *std::find(inferenceTrace.begin(), endSearchIt, edge->getDestination()) << std::endl;

            if (std::find(inferenceTrace.begin(), endSearchIt, edge->getDestination()) != endSearchIt)
                continue;

            nbEvaluatedPrograms++;
            nbExecutedLines += edge->getProgram().getNbLines();
            analyzeProgram(nbExecutionPerInstruction, edge->getProgram());
        }
    }

    this->inferenceTracesStats.push_back({inferenceTrace, nbEvaluatedTeams, nbEvaluatedPrograms, nbExecutedLines, nbExecutionPerInstruction});

    // Update distributions
    this->distribNbEvaluatedTeams[nbEvaluatedTeams]++;
    this->distribNbEvaluatedPrograms[nbEvaluatedPrograms]++;
    this->distribNbExecutedLines[nbExecutedLines]++;
    for (const std::pair<const size_t, const size_t>& p : nbExecutionPerInstruction) {
        this->distribNbExecutionPerInstruction[p.first][p.second]++;
    }
    for (const TPG::TPGVertex* inferenceTraceVertex : inferenceTrace) {
        this->distribNbUsedVertices[inferenceTraceVertex]++;
    }
}

void TPG::ExecutionStats::analyzeExecution(
    const TPG::TPGExecutionEngineInstrumented& tee, const TPG::TPGGraph* graph)
{
    clearInferenceTracesStats();
    this->lastAnalyzedGraph = graph; // Will be used by writeStatsToJson()

    analyzeInstrumentedGraph(graph);

    for (const auto& inferenceTrace : tee.getInferenceTraceHistory()) {
        analyzeInferenceTrace(inferenceTrace);
    }
}

double TPG::ExecutionStats::getAvgEvaluatedTeams() const
{
    return this->avgEvaluatedTeams;
}
double TPG::ExecutionStats::getAvgEvaluatedPrograms() const
{
    return this->avgEvaluatedPrograms;
}
double TPG::ExecutionStats::getAvgExecutedLines() const
{
    return this->avgExecutedLines;
}
const std::map<size_t, double>& TPG::ExecutionStats::
    getAvgNbExecutionPerInstruction() const
{
    return this->avgNbExecutionPerInstruction;
}

const std::vector<TPG::InferenceTraceStats>& TPG::ExecutionStats::getInferenceTracesStats() const
{
    return this->inferenceTracesStats;
}

const std::map<size_t, size_t>& TPG::ExecutionStats::getDistribEvaluatedTeams() const
{
    return this->distribNbEvaluatedTeams;
}
const std::map<size_t, size_t>& TPG::ExecutionStats::getDistribEvaluatedPrograms() const
{
    return this->distribNbEvaluatedPrograms;
}
const std::map<size_t, size_t>& TPG::ExecutionStats::getDistribExecutedLines()
    const
{
    return this->distribNbExecutedLines;
}
const std::map<size_t, std::map<size_t, size_t>>& TPG::ExecutionStats::
    getDistribNbExecutionPerInstruction() const
{
    return this->distribNbExecutionPerInstruction;
}
const std::map<const TPG::TPGVertex*, size_t>& TPG::ExecutionStats::
    getDistribUsedVertices() const
{
    return this->distribNbUsedVertices;
}

void TPG::ExecutionStats::clearInferenceTracesStats()
{
    this->inferenceTracesStats.clear();
    this->distribNbEvaluatedTeams.clear();
    this->distribNbEvaluatedPrograms.clear();
    this->distribNbExecutedLines.clear();
    this->distribNbExecutionPerInstruction.clear();
    this->distribNbUsedVertices.clear();
}

void TPG::ExecutionStats::writeStatsToJson(const char* filePath, bool noIndent) const
{
    std::map<const TPGVertex*, unsigned int> vertexIndexes;
    if (this->lastAnalyzedGraph != nullptr) {
        
        // Store the index of each vertex in the TPGGraph in a lookup table
        // to print the inference traces.
        const std::vector<const TPG::TPGVertex *> graphVertices = this->lastAnalyzedGraph->getVertices();
        for (int i = 0; i < graphVertices.size(); i++) {
            vertexIndexes[graphVertices[i]] = i;
        }
    }

    std::ofstream outputFile(filePath);

    Json::Value root;

    // Average statistics
    root["ExecutionStats"]["avgEvaluatedTeams"] = this->avgEvaluatedTeams;
    root["ExecutionStats"]["avgEvaluatedPrograms"] = this->avgEvaluatedPrograms;
    root["ExecutionStats"]["avgExecutedLines"] = this->avgExecutedLines;

    for (const auto& p : this->avgNbExecutionPerInstruction)
        root["ExecutionStats"]["avgNbExecutionPerInstruction"][std::to_string(p.first)] = p.second;

    // Distributions
    for (const auto& p : this->distribNbEvaluatedTeams)
        root["ExecutionStats"]["distributionNbEvaluatedTeams"][std::to_string(p.first)] = p.second;

    for (const auto& p : this->distribNbEvaluatedPrograms)
        root["ExecutionStats"]["distributionNbEvaluatedPrograms"][std::to_string(p.first)] = p.second;

    for (const auto& p : this->distribNbExecutedLines)
        root["ExecutionStats"]["distributionNbExecutedLines"][std::to_string(p.first)] = p.second;

    for (const auto& p1 : this->distribNbExecutionPerInstruction) {
        for (const auto& p2 : p1.second)
            root["ExecutionStats"]["distributionNbExecutionPerInstruction"][std::to_string(p1.first)][std::to_string(p2.first)] = p2.second;
    }

    for (const auto& p : this->distribNbUsedVertices) {
        size_t idxVertex = vertexIndexes[p.first];
        root["ExecutionStats"]["distributionNbUsedVertices"][std::to_string(idxVertex)] = p.second;
    }

    // Trace statistics
    int i = 0;
    for (TPG::InferenceTraceStats inferenceTraceStats: this->getInferenceTracesStats()) {
        std::string nbTrace = std::to_string(i);

        if (this->lastAnalyzedGraph != nullptr) {
            for (int j = 0; j < inferenceTraceStats.inferenceTrace.size(); j++) {
                root["TracesStats"][nbTrace]["trace"][j] = vertexIndexes[inferenceTraceStats.inferenceTrace[j]];
            }
        }

        root["TracesStats"][nbTrace]["nbEvaluatedTeams"] = inferenceTraceStats.nbEvaluatedTeams;
        root["TracesStats"][nbTrace]["nbEvaluatedPrograms"] = inferenceTraceStats.nbEvaluatedPrograms;
        root["TracesStats"][nbTrace]["nbExecutedLines"] = inferenceTraceStats.nbExecutedLines;
        for (const auto& p : inferenceTraceStats.nbExecutionPerInstruction)
            root["TracesStats"][nbTrace]["nbExecutionPerInstruction"][std::to_string(p.first)] = p.second;
        i++;
    }

    Json::StreamWriterBuilder writerFactory;
    // Set a precision to 6 digits after the point.
    writerFactory.settings_["precision"] = 6U;
    if (noIndent)
        writerFactory.settings_["indentation"] = "";
    Json::StreamWriter* writer = writerFactory.newStreamWriter();
    writer->write(root, &outputFile);
    delete writer;

    outputFile.close();
}
