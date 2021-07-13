#include "data/arrayWrapper.h"

#ifdef __GNUC__
std::string demangle(const char* name)
{
    int status = -4;
    char* demangleValue = abi::__cxa_demangle(name, nullptr, nullptr, &status);
    std::string result;
    if (status == 0) {
        result = std::string{demangleValue};
        free(demangleValue);
    }
    else {
        throw std::runtime_error("Error while trying to demangle the value. "
                                 "Return code of abi::__cxa_demangle is " +
                                 std::to_string(status));
    }
    return result;
}
#endif