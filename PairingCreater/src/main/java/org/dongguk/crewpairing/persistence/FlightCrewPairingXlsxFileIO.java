package org.dongguk.crewpairing.persistence;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.*;
import org.dongguk.common.persistence.AbstractXlsxSolutionFileIO;
import org.dongguk.crewpairing.app.PairingApp;
import org.dongguk.crewpairing.domain.*;
import org.dongguk.crewpairing.domain.factory.DomainFactory;
import org.optaplanner.core.api.score.buildin.hardsoft.HardSoftScore;

import java.io.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;
import java.util.*;

import static java.lang.String.valueOf;

@Slf4j
public class FlightCrewPairingXlsxFileIO extends AbstractXlsxSolutionFileIO<PairingSolution> {

    @Override
    public String getOutputFileExtension() {
        return super.getOutputFileExtension();
    }

    @Override
    public PairingSolution read(File informationXlsxFile) {
        try (InputStream in = new BufferedInputStream(new FileInputStream(informationXlsxFile))) {
            XSSFWorkbook workbook = new XSSFWorkbook(in);
            return new FlightCrewPairingXlsxReader(workbook).read();
        } catch (IOException | RuntimeException e) {
            log.error("{} {}", e.getMessage(), "Input File Error. Please Input File Format");
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }

    public List<Pairing> readPairingList(List<Flight> flightList, File pairingXlsxFile) {
        try (InputStream in = new BufferedInputStream(new FileInputStream(pairingXlsxFile))) {
            XSSFWorkbook workbook = new XSSFWorkbook(in);
            return new FlightCrewPairingXlsxReader(workbook).readPairingSet(flightList);
        } catch (IOException | RuntimeException e) {
            log.error("{} {}", e.getMessage(), "Input File Error. Please Input File Format");
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }

    @Override
    public void write(PairingSolution pairingSolution, File file) {
        new FlightCrewPairingXlsxWriter(pairingSolution).write();
    }

    @Getter
    private static class FlightCrewPairingXlsxReader extends AbstractXlsxReader<PairingSolution, HardSoftScore> {
        private final Map<String, Airport> airportMap = new HashMap<>();

        public FlightCrewPairingXlsxReader(XSSFWorkbook workbook) {
            super(workbook, PairingApp.SOLVER_CONFIG);
        }

        @Override
        public PairingSolution read() {
            readTimeData();
            log.debug("Complete Read Time Data");
            readAircraft();         // 수정
            log.debug("Complete Read Aircraft Data");
            readAirport();          // 수정
            log.debug("Complete Read Airport Data");
            readDeadhead();         // 수정
            log.debug("Complete Read Deadhead Data");
            readFlight();
            log.debug("Complete Read Flight Data");
            return PairingSolution.builder()
                    .aircraftList(DomainFactory.getAircraftList())
                    .airportList(DomainFactory.getAirportList())
                    .flightList(DomainFactory.getFlightList())
                    .pairingList(createEntities()).build();
        }

        public List<Pairing> readPairingSet(List<Flight> inputFlightList) {
            List<Pairing> list = new ArrayList<>();

            nextSheet("Sheet");    // Sheet 고르기
            currentRowIterator.next();              // 주제목 스킵

            while (currentRowIterator.hasNext()) {
                XSSFRow row = (XSSFRow) currentRowIterator.next();

                Iterator<Cell> currentCellIterator =  row.cellIterator();

                int indexCnt = (int) currentCellIterator.next().getNumericCellValue();
                List<Flight> pair = new ArrayList<>();
                while (currentCellIterator.hasNext()) {
                    Cell cell = currentCellIterator.next();
                    if (cell.getCellType() == CellType.BLANK || cell.getCellType() == CellType.STRING) {
                        break;
                    }

                    pair.add(inputFlightList.get((int) cell.getNumericCellValue()));
                }

                list.add(new Pairing(indexCnt, pair, 0));
            }

            log.debug("Complete Read Pairing Data");
            return list;
        }

        private void readTimeData() {
            nextSheet("User_Time");      // Sheet 고르기
            currentRowIterator.next();              // 주제목 스킵
            currentRowIterator.next();              // 빈행  스킵
            currentRowIterator.next();              // 보조제목  스킵
            currentRowIterator.next();              // Header  스킵

            XSSFRow row = (XSSFRow) currentRowIterator.next();

            Pairing.setStaticTime((int) row.getCell(1).getNumericCellValue(),
                    (int) row.getCell(2).getNumericCellValue(),
                    (int) row.getCell(3).getNumericCellValue(),
                    (int) row.getCell(4).getNumericCellValue(),
                    (int) row.getCell(5).getNumericCellValue());

            log.info("Complete Read Time Data");
        }

        private void readAircraft() {
            nextSheet("Program_Cost");    // Sheet 고르기
            currentRowIterator.next();              // 주제목 스킵
            currentRowIterator.next();              // Header 스킵

            int indexCnt = 0;
            while (currentRowIterator.hasNext()) {
                XSSFRow row = (XSSFRow) currentRowIterator.next();

                try {
                    DomainFactory.addAircraft(Aircraft.builder()
                            .id(indexCnt++)
                            .type(row.getCell(0).getStringCellValue())
                            .crewNum((int) row.getCell(1).getNumericCellValue())
                            .flightCost((int) row.getCell(2).getNumericCellValue())
                            .layoverCost((int) row.getCell(3).getNumericCellValue())
                            .quickTurnCost((int) row.getCell(4).getNumericCellValue()).build());
                } catch (IllegalStateException e) {
                    log.info("Finish Read Aircraft File");
                    break;
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        }

        private void readAirport() {
            airportMap.clear();
            nextSheet("User_Hotel");    // Sheet 고르기
            currentRowIterator.next();  // 주제목 스킵
            currentRowIterator.next();  // 빈 행 스킵
            currentRowIterator.next();  // Header 스킵

            int indexCnt = 0;
            while (currentRowIterator.hasNext()) {
                XSSFRow row = (XSSFRow) currentRowIterator.next();

                try {
                    if (row.getCell(0).getStringCellValue().isBlank()) {
                        continue;
                    }

                    airportMap.put(row.getCell(0).getStringCellValue(), Airport.builder()
                            .id(indexCnt++)
                            .name(row.getCell(0).getStringCellValue())
                            .hotelCost((int) row.getCell(1).getNumericCellValue())
                            .deadheadCost(new HashMap<>()).build());
                } catch (IllegalStateException e) {
                    log.info("Finish Read Airport File");
                    break;
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }

            log.info("Complete Read Airport Data");
        }

        private void readDeadhead() {
            nextSheet("User_Deadhead");    // Sheet 고르기
            currentRowIterator.next();  // 주제목 스킵
            currentRowIterator.next();  // 빈 행 스킵
            currentRowIterator.next();  // Header 스킵

            while (currentRowIterator.hasNext()) {
                XSSFRow row = (XSSFRow) currentRowIterator.next();

                try {
                    String origin = row.getCell(0).getStringCellValue();

                    if (origin.isBlank()) {
                        continue;
                    }

                    airportMap.get(origin)
                            .putDeadhead(row.getCell(1).getStringCellValue(),
                                    (int) row.getCell(2).getNumericCellValue());
                } catch (IllegalStateException e) {
                    log.info("Finish Read DeadHead File");
                    break;
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }

            DomainFactory.addAllAirport(airportMap);
            log.info("Complete Read Deadhead Data");
        }

        private void readFlight() {
            nextSheet("User_Flight");    // Sheet 고르기
            currentRowIterator.next();  // 주제목 스킵
            currentRowIterator.next();  // 빈 행 스킵
            currentRowIterator.next();  // Header 스킵

            int indexCnt = 0;
            while (currentRowIterator.hasNext()) {
                XSSFRow row = (XSSFRow) currentRowIterator.next();

                try {
                    DomainFactory.addFlight(Flight.builder()
                            .id(indexCnt++)
                            .serialNumber(row.getCell(0).getStringCellValue())
                            .tailNumber(row.getCell(1).getStringCellValue())
                            .originAirport(DomainFactory.getAirport(row.getCell(2).getStringCellValue()))
                            .originTime(row.getCell(3).getLocalDateTimeCellValue())
                            .destAirport(DomainFactory.getAirport(row.getCell(4).getStringCellValue()))
                            .destTime(row.getCell(5).getLocalDateTimeCellValue())
                            .aircraft(DomainFactory.getAircraft(row.getCell(6).getStringCellValue())).build());
                } catch (IllegalStateException e) {
                    log.info("Finish Read Flight File");
                    break;
                } catch (Exception e) {
                    log.error("{}", e.getMessage());
                    throw new RuntimeException(e);
                }
            }
        }

        private List<Pairing> createEntities() {
            List<Flight> flightList = DomainFactory.getFlightList();

            //초기 페어링 Set 구성 어차피 [solver]가 바꿔버려서 의미 없음 아무것도 안넣으면 오류나서 넣는 것
            List<Pairing> pairingList = new ArrayList<>();
            for (int i = 0; i < flightList.size(); i++) {
                List<Flight> pair = new ArrayList<>();
                pair.add(flightList.get(i));
                pairingList.add(new Pairing(i, pair, 0));
            }

            return pairingList;
        }
    }

        @Getter
        private static class FlightCrewPairingXlsxWriter extends AbstractXlsxWriter<PairingSolution, HardSoftScore> {

            public FlightCrewPairingXlsxWriter(PairingSolution pairingSolution) {
                super(pairingSolution, PairingApp.SOLVER_CONFIG);
            }

            @Override
            public void write() {
                String timeStr = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy_MM_dd_HH_mm_ss"));
//                exportPairingData(timeStr);
//                exportUserData1(timeStr);
                exportVisualData(timeStr);
//                exportUserData2(timeStr);
            }

            private void exportPairingData(String timeStr) {
                String fileName = timeStr + "-pairingData.xlsx";

                try (XSSFWorkbook workbook = new XSSFWorkbook()) {
                    XSSFSheet sheet = workbook.createSheet("Data");

                    List<Pairing> pairingList = solution.getPairingList();

                    //Pairing index 셀 스타일(우측 테두리)
                    CellStyle rightBorder = workbook.createCellStyle();
                    rightBorder.setAlignment(HorizontalAlignment.CENTER);
                    rightBorder.setBorderRight(BorderStyle.THIN);

                    Row row = sheet.createRow(0);
                    Cell cell = row.createCell(0);
                    cell.setCellValue("Pairing Data");

                    //Pairing data 테이블
                    int rowIdx = 1;
                    for(Pairing pairing : pairingList){
                        row = sheet.createRow(rowIdx);
                        cell = row.createCell(0);
                        cell.setCellValue(rowIdx-1);
                        cell.setCellStyle(rightBorder);

                        for(int i=0; i<pairing.getPair().size(); i++){
                            cell = row.createCell(i+1);
                            cell.setCellValue(pairing.getPair().get(i).getId());
                        }
                        rowIdx++;
                    }

                    try (FileOutputStream fo = new FileOutputStream("./data/crewpairing/output/" + fileName)) {
                        workbook.write(fo);
                    }
                }catch (IOException e){
                    e.printStackTrace();
                }

                System.out.println("Create Output File : " + fileName);
            }

            public void exportVisualData(String timeStr) {
                String fileName = timeStr + "-visualData.csv";

                List<Pairing> pairingList = solution.getPairingList();
                //첫 항공기의 출발시간을 기준으로 정렬
                pairingList.removeIf(pairing -> pairing.getPair().isEmpty());
                pairingList.sort(Comparator.comparing(a -> a.getPair().get(0).getOriginTime()));

                //첫 항공기의 출발시간~마지막 항공기의 도착 시간까지 타임 테이블 생성
                StringBuilder text = new StringBuilder();
                LocalDateTime f = pairingList.get(0).getPair().get(0).getOriginTime();
                LocalDateTime firstTime = stripMinutes(f);
                LocalDateTime l = firstTime;

                for (Pairing pairing : pairingList) {
                    for (Flight flight : pairing.getPair()) {
                        l = l.isAfter(flight.getDestTime()) ? l : flight.getDestTime();
                    }
                }
                LocalDateTime lastTime = stripMinutes(l);

                //첫 줄에 날짜 단위 입력
                f = firstTime;
                text.append(",,").append(f).append(",");
                f = f.plusHours(1);
                do {
                    if (f.getHour() == 0) text.append(f);
                    text.append(",");

                    f = f.plusHours(1);
                } while (!f.equals(lastTime));
                text.append("\n");

                //두번째 줄에 시간 단위 입력
                text.append("INDEX,TYPE,");
                f = firstTime;
                do {
                    text.append(f.getHour());
                    text.append(":00,");

                    f = f.plusHours(1);
                } while (!f.equals(lastTime));
                text.append("\n");

                //타임 테이블의 내용 작성
                for (Pairing pairing : pairingList) {
                    text.append("SET").append(pairingList.indexOf(pairing)).append(",");
                    text.append(pairing.getPair().get(0).getAircraft().getType()).append(",");
                    String s = buildTable(pairing.getPair(), firstTime);
                    text.append(s);
                }

                //csv 파일로 출력
                try (FileWriter fw = new FileWriter("./data/crewpairing/output/" + fileName)) {
                    fw.write(text.toString());
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }

            //출발시간과 도착 시간의 차이를 구하며 csv format 에 맞는 text 생성.
            private static String buildTable(List<Flight> pairing, LocalDateTime firstTime) {
                StringBuilder sb = new StringBuilder();
                for (Flight flight : pairing) {
                    int a = (int) ChronoUnit.HOURS.between(firstTime, stripMinutes(flight.getOriginTime()));
                    sb.append(",".repeat(Math.max(0, a)));
                    sb.append(flight.getOriginAirport().getName());
                    sb.append(",");
                    int b = (int) ChronoUnit.HOURS.between(flight.getOriginTime(), stripMinutes(flight.getDestTime()));
                    sb.append("#######,".repeat(Math.max(0, b - 1)));
                    sb.append(flight.getDestAirport().getName());

                    firstTime = stripMinutes(flight.getDestTime());
                }
                sb.append("\n");

                return valueOf(sb);
            }

            //분 단위를 버림함
            private static LocalDateTime stripMinutes(LocalDateTime l) {
                return LocalDateTime.of(l.getYear(), l.getMonth(), l.getDayOfMonth(), l.getHour(), 0);
            }
        }
    }
