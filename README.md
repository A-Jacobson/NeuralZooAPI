# ZooAPI
Machine Learning API for DL models


## Milestones
- [x] Example endpoint with torchvision models
- [ ] Depth in the Wild
- [ ] Super Resolution
- [ ] CoNLL Entity Detection
- [ ] Amazon Reviews Sentiment Analysis
- [ ] Word2Vec
- [ ] Dockerize
- [ ] Deploy to AWS

## Endpoints
![title](api_spec.png)

# To Run
```bash
# build dockerfile
docker build -t neuralzoo .
docker run -p 4000:80 neuralzoo

# optionally, run in headless mode
docker run -d -p 4000:80 neuralzoo
 
# to stop
docker container ls # find CONTAINER ID
docker container stop <CONTAINER ID>

```
