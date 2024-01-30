import fs from 'fs';
import { layoutProcess } from 'bpmn-auto-layout';

const path = './tmp/foo.bpmn';

fs.readFile(path, 'utf8', async (err, diagramXML) => {
    if (err) {
        console.error('Error reading BPMN XML file:', err);
        return;
    }

    try {
        // Layout the BPMN diagram
        const layoutedDiagramXML = await layoutProcess(diagramXML);

        // Log the layouted XML
        console.log(layoutedDiagramXML);
    } catch (layoutError) {
        console.error('Error during BPMN layout:', layoutError);
    }
});
