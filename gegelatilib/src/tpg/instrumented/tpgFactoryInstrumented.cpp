/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2022) :
 *
 * Karol Desnos <kdesnos@insa-rennes.fr> (2022)
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

#include "tpg/instrumented/tpgFactoryInstrumented.h"
#include "tpg/instrumented/tpgActionInstrumented.h"
#include "tpg/instrumented/tpgEdgeInstrumented.h"
#include "tpg/instrumented/tpgExecutionEngineInstrumented.h"
#include "tpg/instrumented/tpgTeamInstrumented.h"

std::shared_ptr<TPG::TPGGraph> TPG::TPGFactoryInstrumented::createTPGGraph(
    const Environment& env) const
{
    return std::make_shared<TPG::TPGGraph>(
        env, std::make_unique<TPGFactoryInstrumented>());
}

TPG::TPGTeam* TPG::TPGFactoryInstrumented::createTPGTeam() const
{
    return new TPGTeamInstrumented();
}

TPG::TPGAction* TPG::TPGFactoryInstrumented::createTPGAction(
    const uint64_t id) const
{
    return new TPGActionInstrumented(id);
}

std::unique_ptr<TPG::TPGEdge> TPG::TPGFactoryInstrumented::createTPGEdge(
    const TPGVertex* src, const TPGVertex* dest,
    const std::shared_ptr<Program::Program> prog) const
{
    auto ptr = std::make_unique<TPG::TPGEdgeInstrumented>(src, dest, prog);
    return ptr;
}

std::unique_ptr<TPG::TPGExecutionEngine> TPG::TPGFactoryInstrumented::
    createTPGExecutionEngine(const Environment& env, Archive* arch) const
{
    return std::make_unique<TPGExecutionEngineInstrumented>(env, arch);
}

void TPG::TPGFactoryInstrumented::resetTPGGraphCounters(
    const TPG::TPGGraph& tpg) const
{
    // Reset all vertices
    for (const TPG::TPGVertex* vertex : tpg.getVertices()) {
        const TPG::TPGVertexInstrumented* vertexI =
            dynamic_cast<const TPG::TPGVertexInstrumented*>(vertex);
        if (vertexI != nullptr) {
            vertexI->reset();
        }
    }

    // Reset all edges
    for (const auto& edge : tpg.getEdges()) {
        const TPG::TPGEdgeInstrumented* edgeI =
            dynamic_cast<const TPG::TPGEdgeInstrumented*>(edge.get());
        if (edgeI != nullptr) {
            edgeI->reset();
        }
    }
}

void TPG::TPGFactoryInstrumented::clearUnusedTPGGraphElements(
    TPG::TPGGraph& tpg) const
{
    // Remove unused vertices first
    // (this will remove a few edges as a side-effect)
    // Work on a copy of vertex list as the graph is modified during the for
    // loop.
    std::vector<const TPG::TPGVertex*> vertices(tpg.getVertices());
    for (const TPG::TPGVertex* vertex : vertices) {
        const TPG::TPGVertexInstrumented* vertexI =
            dynamic_cast<const TPG::TPGVertexInstrumented*>(vertex);
        // If the vertex is instrumented AND was never visited
        if (vertexI != nullptr && vertexI->getNbVisits() == 0) {
            // remove it
            tpg.removeVertex(*vertex);
        }
    }

    // Remove un-traversed edges
    std::vector<const TPG::TPGEdge*> edges;
    // Copy the edge list before iteration
    for (auto& edge : tpg.getEdges()) {
        edges.push_back(edge.get());
    }
    // Iterate on the edge list
    for (auto edge : edges) {
        const TPG::TPGEdgeInstrumented* edgeI =
            dynamic_cast<const TPG::TPGEdgeInstrumented*>(edge);
        if (edgeI != nullptr && edgeI->getNbTraversal() == 0) {
            tpg.removeEdge(*edge);
        }
    }
}

void TPG::TPGFactoryInstrumented::clearUnusedTPGGraphElementsV2(
    TPG::TPGGraph& tpg) const
{
    // Remove unused vertices first
    // (this will remove a few edges as a side-effect)
    // Work on a copy of vertex list as the graph is modified during the for
    // loop.
    std::vector<const TPG::TPGVertex*> vertices(tpg.getVertices());
    for (const TPG::TPGVertex* vertex : vertices) {
        const TPG::TPGVertexInstrumented* vertexI =
            dynamic_cast<const TPG::TPGVertexInstrumented*>(vertex);
        // If the vertex is instrumented AND was never visited
        if (vertexI != nullptr && vertexI->getNbVisits() == 0) {
            // remove it
            tpg.removeVertex(*vertex);
        }
    }

    // Remove un-traversed edges
    std::vector<const TPG::TPGEdge*> edges;
    // Copy the edge list before iteration
    for (auto& edge : tpg.getEdges()) {
        edges.push_back(edge.get());
    }
    // Iterate on the edge list
    for (auto edge : edges) {
        const TPG::TPGEdgeInstrumented* edgeI =
            dynamic_cast<const TPG::TPGEdgeInstrumented*>(edge);
        if (edgeI != nullptr && edgeI->getNbTraversal() == 0) {
            tpg.removeEdge(*edge);
        }
    }

    // TPG graph is now cleared of all unvisited vertices and untraversed edges.
    // Now we delete vertices that have a single outgoing edge and replace them
    // by the following vertex of the edge. 
    // The edge leading to the deleted vertex is untouched since it is the one
    // deciding if we traverse the vertex or not.
    
    // Work on a copy of the vertex list because we will modify the graph.
    {
        std::vector<const TPG::TPGVertex*> vertices(tpg.getVertices());
        for (const TPG::TPGVertex* vertex : vertices) {
            // Rebuild a snapshot of edges for safe iteration while modifying the graph.
            std::vector<const TPG::TPGEdge*> edgesSnapshot;
            for (const auto& ePtr : tpg.getEdges()) {
                edgesSnapshot.push_back(ePtr.get());
            }

            // Collect outgoing edges from 'vertex'
            std::vector<const TPG::TPGEdge*> outgoing;
            for (const TPG::TPGEdge* e : edgesSnapshot) {
                if (e->getSource() == vertex) {
                    outgoing.push_back(e);
                }
            }

            // Only target vertices with exactly one outgoing edge
            if (outgoing.size() != 1) {
                continue;
            }

            const TPG::TPGEdge* outEdge = outgoing.front();
            const TPG::TPGVertex* succ = outEdge->getDestination();
            if (succ == nullptr) {
                // print error case, should not happen in a well-formed TPG graph
                std::cerr << "Error: vertex " << vertex
                          << " has an outgoing edge with null destination."
                          << std::endl;
                continue;
            }

            // Gather incoming edges that point to 'vertex'
            std::vector<const TPG::TPGEdge*> incoming;
            for (const TPG::TPGEdge* e : edgesSnapshot) {
                if (e->getDestination() == vertex) {
                    incoming.push_back(e);
                }
            }

            // For each incoming edge, create a new edge from the same source to the successor,
            // then remove the old incoming edge. We skip self-loops from vertex -> vertex.
            for (const TPG::TPGEdge* inE : incoming) {
                const TPG::TPGVertex* src = inE->getSource();
                if (src == nullptr || src == vertex) {
                    // print error self-loop or malformed edge
                    std::cerr << "Error: vertex " << vertex
                              << " has an incoming edge with null source or self-loop."
                              << std::endl;
                    continue;
                }

                // // Duplicate the program pointer used by the incoming edge
                // std::shared_ptr<Program::Program> prog = inE->getProgramSharedPointer();

                // // Add redirected edge src -> succ
                // tpg.addNewEdge(*src, *succ, prog);

                // // Remove the old incoming edge
                // tpg.removeEdge(*inE);
                    
                // rewire the incoming edge to point to the successor instead of the deleted vertex
                tpg.setEdgeDestination(*inE, *succ);
            }

            // remove the outgoing edge
            tpg.removeEdge(*outEdge);

            // remove vertex
            tpg.removeVertex(*vertex);

        // // After rewiring incoming edges, remove all edges attached to 'vertex' (including its unique outgoing edge)
        // // Build a fresh snapshot to find edges to remove safely.
        // std::vector<const TPG::TPGEdge*> toRemove;
        // for (const auto& ePtr : tpg.getEdges()) {
        //     const TPG::TPGEdge* e = ePtr.get();
        //     if (e->getSource() == vertex || e->getDestination() == vertex) {
        //         toRemove.push_back(e);
        //     }
        // }
        // for (const TPG::TPGEdge* e : toRemove) {
        //     tpg.removeEdge(*e);
        // }

        // // Finally remove the vertex itself
        // tpg.removeVertex(*vertex);

        }
    }
}
