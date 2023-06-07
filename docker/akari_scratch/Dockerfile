FROM node:16.20.0-alpine3.16

COPY scratch-vm /resources/scratch-vm
COPY scratch-gui /resources/scratch-gui

WORKDIR /resources/scratch-vm
RUN yarn install
RUN yarn link


WORKDIR /resources/scratch-gui
RUN yarn link scratch-vm
RUN yarn install

EXPOSE 52001
CMD ["yarn", "start"]
