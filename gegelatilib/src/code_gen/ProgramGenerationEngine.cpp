/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2019) :
 *
 * Karol Desnos <kdesnos@insa-rennes.fr> (2019)
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

#include "code_gen/ProgramGenerationEngine.h"
#include "data/printableDataHandler.h"

const std::regex CodeGen::ProgramGenerationEngine::operand_regex("(\\$[0-9]*)");
const std::string CodeGen::ProgramGenerationEngine::nameRegVariable("reg");
const std::string CodeGen::ProgramGenerationEngine::nameDataVariable("in");
const std::string CodeGen::ProgramGenerationEngine::nameOperandVariable("op");

void CodeGen::ProgramGenerationEngine::generateCurrentLine() {
    // const Program::Line& line = this->getCurrentLine();
    const Instructions::Instruction &instruction =
            this->getCurrentInstruction();
    auto prtIns =
            dynamic_cast<const Instructions::PrintableInstruction *>(&instruction);

    if (prtIns != nullptr) {
        // std::map<operand(uint64t ?), uint_64t> initOperand // = (operand,
        // chiffre après "op")
        fileC << "\t{" << std::endl;
        initOperandCurrentLine();
        std::string codeLine = completeFormat(*prtIns);
        // init
        fileC << "\t\t" << codeLine << "\n"
              << "\t}" << std::endl;
    } else {
        throw std::runtime_error("The pointer on the instruction cannot be"
                                 " converted to a pointer on a printable"
                                 "instruction.");
    }
}

void CodeGen::ProgramGenerationEngine::generateProgram(
        uint64_t progID, const bool ignoreException) {
    fileC << "\ndouble P" << progID << "(){" << std::endl;
    fileH << "double P" << progID << "();" << std::endl;

    // instantiate register
    fileC << "\tdouble " << nameRegVariable << "["
          << program->getEnvironment().getNbRegisters() << "] = {0";
    for (int i = 1; i < program->getEnvironment().getNbRegisters(); ++i) {
        fileC << ", 0";
    }
    fileC << "};" << std::endl;
    this->programCounter = 0;
    // todo embarqué dans une fonction globale avec une operateur() ? avec un
    // objet foncteur
    //  Iterate over the lines of the Program
    bool hasNext = this->program->getNbLines() > 0;

    // Skip first lines if they are introns.
    if (hasNext && this->program->isIntron(0)) {
        hasNext = this->next();
    }

    // Execute useful lines
    while (hasNext) {
        try {
            // generate the current line
            this->generateCurrentLine(); // todo utilisez un objet foncteur à la
            // place
        }
        catch (std::out_of_range e) {
            if (!ignoreException) {
                throw; // rethrow
            }
        }

        // Increment the programCounter.
        hasNext = this->next();
    };
#ifdef DEBUG
    fileC << "#ifdef DEBUG" << std::endl;
    fileC << "\tprintf(\"P" << progID << " : reg[0] = %lf \\n\", reg[0]);"
          << std::endl;
    fileC << "#endif" << std::endl;
#endif
    fileC << "\treturn reg[0];\n}" << std::endl;
}

std::string CodeGen::ProgramGenerationEngine::completeFormat(
        const Instructions::PrintableInstruction &instruction) const {
    const std::string &format = instruction.getFormat();
    const Program::Line &line =
            this->getCurrentLine(); // throw std::out_of_range
    std::string codeLine(format);
    std::string operandValue;
    for (auto itr =
            std::sregex_iterator(format.begin(), format.end(), operand_regex);
         itr != std::sregex_iterator(); ++itr) {
        const std::string &match = (*itr).str();
        auto pos = codeLine.find(match);
        // get number after character '$'
        int idx = std::stoi(match.substr(1));
        if (idx > 0) {
            const uint64_t operandLocation = this->getOperandLocation(idx - 1);
            std::string operandIdx(std::to_string(idx - 1));
            operandValue = nameOperandVariable + operandIdx;
            // if number > 0 means that it's in the left side of the operation
//            if (line.getOperand(idx - 1).first == 0) {
//                // operand value is an intern register
//                operandValue = nameOperandVariable + "[" +
//                               std::to_string(operandLocation) + "]";
//            } else {
//                // operandValue come from the environment
//                operandValue = nameDataVariable +
//                               std::to_string(line.getOperand(idx - 1).first) +
//                               "[" + std::to_string(operandLocation) + "]";
//            }
        } else {
            // if number == 0 it corresponds to the result of the function
            operandValue = nameRegVariable + "[" +
                           std::to_string(line.getDestinationIndex()) + "]";
        }
        codeLine.replace(pos, match.size(), operandValue);
    }
    return codeLine;
}

void CodeGen::ProgramGenerationEngine::initGlobalVar() {

    for (int i = 1; i < this->dataScsConstsAndRegs.size(); ++i) {

        const Data::DataHandler &d = this->dataScsConstsAndRegs.at(i);
        auto printableDataSource =
                dynamic_cast<const Data::PrintableDataHandler *>(&d);
        //todo fonction de cast pour pas dupliquer du code ?
        if (printableDataSource == nullptr) {
            throw std::runtime_error("The data source cannot be"
                                     " converted to a pointer on a printable"
                                     "instruction.");
        }
        std::string type = printableDataSource->getTemplatedType();
        fileC << "extern " << type << "* in" << i << ";" << std::endl;
    }
}

void CodeGen::ProgramGenerationEngine::openFile(const std::string &filename,
                                                const std::string &path) {
    if (filename.size() == 0) {
        std::cout << "filename is empty" << std::endl;
        throw std::invalid_argument("filename is empty");
    }
    try {
        this->fileC.open(path + filename + ".c", std::ofstream::out);
        this->fileH.open(path + filename + ".h", std::ofstream::out);
    }
    catch (std::ios_base::failure e) {
        throw std::runtime_error("Could not open file " +
                                 std::string(path + filename));
    }
    fileC << "#include \"" << filename << ".h\"" << std::endl;
    initGlobalVar();
    fileH << "#ifndef C_" << filename << "_H" << std::endl;
    fileH << "#define C_" << filename << "_H\n" << std::endl;
    fileC << "#include \"externHeader.h\"" << std::endl;
#ifdef DEBUG
    fileC << "#include <stdio.h>" << std::endl;
#endif // DEBUG
}

void CodeGen::ProgramGenerationEngine::initOperandCurrentLine() {
    uint64_t opIdx;
    const Program::Line &line = getCurrentLine();
    const Instructions::Instruction &instruction = getCurrentInstruction();
    auto printableIns = dynamic_cast<const Instructions::PrintableInstruction *>(&instruction);
    if (printableIns == nullptr){
        throw std::runtime_error("The pointer on the instruction cannot be"
                                 " converted to a pointer on a printable"
                                 "instruction.");
    }
    for (int i = 0; i < instruction.getNbOperands(); ++i) {
        uint64_t sourceIdx = line.getOperand(i).first;
        const std::type_info &operandType =
                instruction.getOperandTypes().at(i).get();
        opIdx = this->getOperandLocation(i);
        const Data::DataHandler &dataSource = this->dataScsConstsAndRegs.at(
                sourceIdx); // Throws std::out_of_range
        auto printableDataSource =
                dynamic_cast<const Data::PrintableDataHandler *>(&dataSource);
        if (printableDataSource == nullptr) {
            throw std::runtime_error(
                    "error can't cast dataHandler into a PrintableDataHandler\n");
        }
        fileC << "\t\t" << printableIns->getPrimitiveType(i) << " "
                                                << nameOperandVariable << i;
        std::vector<uint64_t> vectIdx =
                printableDataSource->getDataIndexes(operandType, opIdx);

        size_t size = vectIdx.size();
        if (size <= 0) {
            throw std::runtime_error("Vector of indexes is negative or equal "
                                     "to 0, can't generate the "
                                     "declaration of the variable\n");
        }
        if (size == 1) {
            fileC << " = ";
            printNameSourceData(i);
            fileC << "[" << vectIdx.at(0) << "];\n";
        } else {
            fileC << "[] = {";
            for (int j = 0; j < size; ++j) {
                printNameSourceData(i);
                fileC << "[" << vectIdx.at(j) << "]";
                if (j < (size - 1)) {
                    fileC << ",";
                }
            }
            fileC << "};\n";
        }
    }
}

void CodeGen::ProgramGenerationEngine::printInitOperand(
        const std::vector<uint64_t> &vectIdx) {
}

void CodeGen::ProgramGenerationEngine::printNameSourceData(const uint64_t idx) {
    uint64_t dataSourceIdx = getCurrentLine().getOperand(idx).first;
    if (dataSourceIdx == 0) {
        fileC << nameRegVariable;
    } else {
        fileC << nameDataVariable << dataSourceIdx;
    }
}

#endif // CODE_GENERATION