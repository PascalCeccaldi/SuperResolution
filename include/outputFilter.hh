#ifndef OUTPUTFILTER_HH_
# define OUTPUTFILTER_HH_

# include <string>
# include "tbb/pipeline.h"
# include "cv.h"
# include "highgui.h"

class OutputFilter : public tbb::filter
{
    public:
        OutputFilter(CvVideoWriter* writer);
        std::string get_name();
    private:
        void* operator()(void*);
        CvVideoWriter* writer;
        int counter = 0;
};

#endif /* !OUTPUTFILTER_HH_ */
