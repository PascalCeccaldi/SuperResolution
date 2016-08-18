#ifndef INPUTFILTER_HH_
# define INPUTFILTER_HH_

# include <string>
# include "tbb/pipeline.h"
# include "cv.h"
# include "highgui.h"

class InputFilter : public tbb::filter
{
    public:
        InputFilter(CvCapture* cap);
        std::string get_name();
    private:
        void* operator()(void*);
        CvCapture* capture;
};

#endif /* !INPUTFILTER_HH_ */
